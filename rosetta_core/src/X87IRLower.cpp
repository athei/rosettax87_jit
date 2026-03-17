#include "rosetta_core/X87IR.h"

#include <cstring>
#include <utility>

#include "rosetta_core/AssemblerHelpers.hpp"
#include "rosetta_core/IROperand.h"
#include "rosetta_core/Register.h"
#include "rosetta_core/TranslationResult.h"
#include "rosetta_core/TranslatorHelpers.hpp"
#include "rosetta_core/TranslatorX87Helpers.hpp"
#include "rosetta_core/X87Cache.h"

// Internal helpers from TranslatorX87Internal.hpp that we need.
namespace TranslatorX87 {
inline auto x87_begin(TranslationResult& a1, AssemblerBuffer& buf) -> std::pair<int, int> {
    if (a1.x87_cache.run_remaining > 0 && a1.x87_cache.gprs_valid)
        return {a1.x87_cache.base_gpr, a1.x87_cache.top_gpr};
    const int Xbase = alloc_gpr(a1, 0);
    const int Wd_top = alloc_gpr(a1, 1);
    emit_x87_base(buf, a1, Xbase);
    emit_load_top(buf, a1, Xbase, Wd_top);
    if (a1.x87_cache.run_remaining > 0) {
        a1.x87_cache.base_gpr = static_cast<int8_t>(Xbase);
        a1.x87_cache.top_gpr = static_cast<int8_t>(Wd_top);
        const int Xst_base = alloc_gpr(a1, 6);
        emit_add_imm(buf, 1, 0, 0, 0, kX87RegFileOff, Xbase, Xst_base);
        a1.x87_cache.st_base_gpr = static_cast<int8_t>(Xst_base);
        a1.x87_cache.gprs_valid = 1;
    }
    return {Xbase, Wd_top};
}
inline int x87_get_st_base(TranslationResult& a1) {
    return a1.x87_cache.gprs_valid ? a1.x87_cache.st_base_gpr : -1;
}
}  // namespace TranslatorX87

namespace X87IR {

static constexpr int16_t kX87TagWordImm12 = kX87TagWordOff / 2;  // = 2

// ── FPR assignment ──────────────────────────────────────────────────────────

struct FPRState {
    int8_t node_fpr[kMaxNodes];      // D register for each node, or -1
    int16_t last_use[kMaxNodes];     // last node index that uses each node, or -1

    void compute_last_uses(const Context& ctx) {
        memset(last_use, -1, sizeof(last_use));
        memset(node_fpr, -1, sizeof(node_fpr));
        for (int i = 0; i < ctx.num_nodes; i++) {
            auto& n = ctx.nodes[i];
            if (n.flags & kDead) continue;
            for (int j = 0; j < 3; j++) {
                if (n.inputs[j] >= 0) last_use[n.inputs[j]] = static_cast<int16_t>(i);
            }
        }
        // Final stack values are live past all nodes.
        for (int d = 0; d < 8; d++) {
            int16_t val = ctx.slot_val[d];
            if (val >= 0 && val < ctx.num_nodes)
                last_use[val] = static_cast<int16_t>(ctx.num_nodes);
        }
    }

    // Free FPRs whose last use was node `i`.
    void free_dead_inputs(TranslationResult& result, const Node& n, int i) {
        for (int j = 0; j < 3; j++) {
            int16_t in = n.inputs[j];
            if (in >= 0 && last_use[in] == i && node_fpr[in] >= 0) {
                free_fpr(result, node_fpr[in]);
                node_fpr[in] = -1;
            }
        }
    }

    int get(int16_t node_id) const {
        if (node_id < 0 || node_id >= kMaxNodes) return -1;
        return node_fpr[node_id];
    }

    // Try to reuse an FPR from a dying input of node `i`.
    // Returns the FPR register number, or -1 if no reuse is possible.
    // On success, sets node_fpr[input] = -1 so free_dead_inputs skips it.
    // Caller MUST capture input FPRs via get() BEFORE calling this.
    int try_reuse_input(const Context& ctx, int i) {
        const auto& n = ctx.nodes[i];
        // Prefer inputs[0] (Dn, natural accumulator position)
        for (int pref = 0; pref < 3; pref++) {
            int16_t in = n.inputs[pref];
            if (in >= 0 && last_use[in] == i && node_fpr[in] >= 0) {
                int fpr = node_fpr[in];
                node_fpr[in] = -1;  // claimed — free_dead_inputs will skip
                return fpr;
            }
        }
        return -1;
    }
};

// ── Main lowering ───────────────────────────────────────────────────────────

void lower(Context& ctx, TranslationResult* result) {
    auto& buf = result->insn_buf;

    // ── Preamble: acquire base, TOP, and st_base ────────────────────────────
    auto [Xbase, Wd_top] = TranslatorX87::x87_begin(*result, buf);
    int Xst_base = TranslatorX87::x87_get_st_base(*result);
    int Wd_tmp = alloc_gpr(*result, 2);

    // ── FPR assignment ──────────────────────────────────────────────────────
    FPRState fprs;
    fprs.compute_last_uses(ctx);

    // ── Emit each IR node ───────────────────────────────────────────────────
    for (int i = 0; i < ctx.num_nodes; i++) {
        auto& n = ctx.nodes[i];
        if (n.flags & kDead) continue;

        switch (n.op) {

        // ── Value nodes ─────────────────────────────────────────────────
        case Op::ReadSt: {
            int Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_load_st(buf, Xbase, Wd_top, n.initial_depth, Wd_tmp, Dd, Xst_base);
            break;
        }
        case Op::LoadF64: {
            int Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            int addr = compute_operand_address(*result, true, n.mem_operand, GPR::XZR);
            emit_fldr_imm(buf, 3, Dd, addr, 0);
            free_gpr(*result, addr);
            break;
        }
        case Op::LoadF32: {
            int Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            int addr = compute_operand_address(*result, true, n.mem_operand, GPR::XZR);
            emit_fldr_imm(buf, 2, Dd, addr, 0);
            free_gpr(*result, addr);
            emit_fcvt_s_to_d(buf, Dd, Dd);
            break;
        }
        case Op::LoadI16: {
            int Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            int addr = compute_operand_address(*result, true, n.mem_operand, GPR::XZR);
            int Wd_val = alloc_free_gpr(*result);
            // LDRSH Wd_val, [addr] — sign-extending load
            emit_ldrs(buf, /*is_64=*/0, /*size=S16*/1, Wd_val, addr);
            free_gpr(*result, addr);
            // SCVTF Dd, Wd_val
            emit_scvtf(buf, /*is_64_int=*/0, /*ftype=f64*/1, Dd, Wd_val);
            free_gpr(*result, Wd_val);
            break;
        }
        case Op::LoadI32: {
            int Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            int addr = compute_operand_address(*result, true, n.mem_operand, GPR::XZR);
            int Wd_val = alloc_free_gpr(*result);
            emit_ldr_imm(buf, /*size=S32*/2, Wd_val, addr, 0);
            free_gpr(*result, addr);
            // SXTW + SCVTF: sign-extend 32→64 then convert
            emit_scvtf(buf, /*is_64_int=*/0, /*ftype=f64*/1, Dd, Wd_val);
            free_gpr(*result, Wd_val);
            break;
        }
        case Op::LoadI64: {
            int Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            int addr = compute_operand_address(*result, true, n.mem_operand, GPR::XZR);
            int Xd_val = alloc_free_gpr(*result);
            emit_ldr_imm(buf, /*size=S64*/3, Xd_val, addr, 0);
            free_gpr(*result, addr);
            emit_scvtf_x_to_d(buf, Dd, Xd_val);
            free_gpr(*result, Xd_val);
            break;
        }
        case Op::ConstZero: {
            int Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_movi_d_zero(buf, Dd);
            break;
        }
        case Op::ConstOne: {
            int Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_fmov_d_one(buf, Dd);
            break;
        }
        case Op::ConstF64: {
            int Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_ldr_literal_f64(buf, Dd, n.imm_bits);
            break;
        }

        // ── Binary arithmetic ───────────────────────────────────────────
        case Op::FAdd: {
            int Dn = fprs.get(n.inputs[0]), Dm = fprs.get(n.inputs[1]);
            int Dd = fprs.try_reuse_input(ctx, i);
            if (Dd < 0) Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_fadd_f64(buf, Dd, Dn, Dm);
            break;
        }
        case Op::FSub: {
            int Dn = fprs.get(n.inputs[0]), Dm = fprs.get(n.inputs[1]);
            int Dd = fprs.try_reuse_input(ctx, i);
            if (Dd < 0) Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_fsub_f64(buf, Dd, Dn, Dm);
            break;
        }
        case Op::FMul: {
            int Dn = fprs.get(n.inputs[0]), Dm = fprs.get(n.inputs[1]);
            int Dd = fprs.try_reuse_input(ctx, i);
            if (Dd < 0) Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_fmul_f64(buf, Dd, Dn, Dm);
            break;
        }
        case Op::FDiv: {
            int Dn = fprs.get(n.inputs[0]), Dm = fprs.get(n.inputs[1]);
            int Dd = fprs.try_reuse_input(ctx, i);
            if (Dd < 0) Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_fdiv_f64(buf, Dd, Dn, Dm);
            break;
        }

        // ── FMA ─────────────────────────────────────────────────────────
        case Op::FMAdd: {
            // FMADD Dd, Dn, Dm, Da → Da + Dn * Dm
            // inputs[0] = Dn, inputs[1] = Dm, inputs[2] = Da
            int Dn = fprs.get(n.inputs[0]), Dm = fprs.get(n.inputs[1]);
            int Da = fprs.get(n.inputs[2]);
            int Dd = fprs.try_reuse_input(ctx, i);
            if (Dd < 0) Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_fmadd_f64(buf, Dd, Dn, Dm, Da);
            break;
        }
        case Op::FMSub: {
            // FMSUB Dd, Dn, Dm, Da → Da - Dn * Dm
            int Dn = fprs.get(n.inputs[0]), Dm = fprs.get(n.inputs[1]);
            int Da = fprs.get(n.inputs[2]);
            int Dd = fprs.try_reuse_input(ctx, i);
            if (Dd < 0) Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_fmsub_f64(buf, Dd, Dn, Dm, Da);
            break;
        }
        case Op::FNMSub: {
            // FNMSUB Dd, Dn, Dm, Da → Dn * Dm - Da
            int Dn = fprs.get(n.inputs[0]), Dm = fprs.get(n.inputs[1]);
            int Da = fprs.get(n.inputs[2]);
            int Dd = fprs.try_reuse_input(ctx, i);
            if (Dd < 0) Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_fnmsub_f64(buf, Dd, Dn, Dm, Da);
            break;
        }

        // ── Unary ───────────────────────────────────────────────────────
        case Op::FNeg: {
            int Dn = fprs.get(n.inputs[0]);
            int Dd = fprs.try_reuse_input(ctx, i);
            if (Dd < 0) Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_fneg_f64(buf, Dd, Dn);
            break;
        }
        case Op::FAbs: {
            int Dn = fprs.get(n.inputs[0]);
            int Dd = fprs.try_reuse_input(ctx, i);
            if (Dd < 0) Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_fabs_f64(buf, Dd, Dn);
            break;
        }
        case Op::FSqrt: {
            int Dn = fprs.get(n.inputs[0]);
            int Dd = fprs.try_reuse_input(ctx, i);
            if (Dd < 0) Dd = alloc_free_fpr(*result);
            fprs.node_fpr[i] = static_cast<int8_t>(Dd);
            emit_fsqrt_f64(buf, Dd, Dn);
            break;
        }

        // ── Memory stores ───────────────────────────────────────────────
        case Op::StoreF64: {
            int addr = compute_operand_address(*result, true, n.mem_operand, GPR::XZR);
            emit_fstr_imm(buf, 3, fprs.get(n.inputs[0]), addr, 0);
            free_gpr(*result, addr);
            break;
        }
        case Op::StoreF32: {
            // Narrow f64 → f32, then store.
            int Ds_tmp = alloc_free_fpr(*result);
            emit_fcvt_d_to_s(buf, Ds_tmp, fprs.get(n.inputs[0]));
            int addr = compute_operand_address(*result, true, n.mem_operand, GPR::XZR);
            emit_fstr_imm(buf, 2, Ds_tmp, addr, 0);
            free_gpr(*result, addr);
            free_fpr(*result, Ds_tmp);
            break;
        }

        // ── Compare ─────────────────────────────────────────────────────
        case Op::FCmp: {
            int Wd_save = alloc_free_gpr(*result);
            emit_mrs_nzcv(buf, Wd_save);
            emit_fcmp_f64(buf, fprs.get(n.inputs[0]), fprs.get(n.inputs[1]));

            int Wd_packed = alloc_free_gpr(*result);
            emit_fcom_cc_pack(buf, *result, Wd_packed, Wd_save);
            // emit_fcom_cc_pack restores NZCV and frees Wd_save internally.

            if (n.flags & kFcomFused) {
                // Fused: keep packed CC alive for FStsw to consume.
                // Store the GPR number in node_fpr[] (repurposed for GPR tracking).
                fprs.node_fpr[i] = static_cast<int8_t>(Wd_packed);
            } else {
                // Non-fused: write CC to status_word now.
                emit_fcom_cc_write_sw(buf, *result, Xbase, Wd_packed);
                free_gpr(*result, Wd_packed);
            }
            break;
        }
        case Op::FTst: {
            int Wd_save = alloc_free_gpr(*result);
            emit_mrs_nzcv(buf, Wd_save);
            emit_fcmp_zero_f64(buf, fprs.get(n.inputs[0]));

            int Wd_packed = alloc_free_gpr(*result);
            emit_fcom_cc_pack(buf, *result, Wd_packed, Wd_save);

            if (n.flags & kFcomFused) {
                fprs.node_fpr[i] = static_cast<int8_t>(Wd_packed);
            } else {
                emit_fcom_cc_write_sw(buf, *result, Xbase, Wd_packed);
                free_gpr(*result, Wd_packed);
            }
            break;
        }

        // ── FSTSW AX ───────────────────────────────────────────────────
        case Op::FStsw: {
            static constexpr int16_t kSwImm12 = kX87StatusWordOff / 2;  // = 1

            if (n.flags & kFcomFused) {
                // Fused: retrieve packed CC from the FCmp node.
                int16_t fcmp_id = n.inputs[0];
                int Wd_packed = fprs.get(fcmp_id);

                // RMW status_word with packed CC bits.
                emit_fcom_cc_write_sw(buf, *result, Xbase, Wd_packed);
                free_gpr(*result, Wd_packed);
                fprs.node_fpr[fcmp_id] = -1;
            }
            // Both paths: load status_word (which now has correct CC), BFI into AX.
            {
                int Wd_sw = alloc_free_gpr(*result);
                emit_ldr_str_imm(buf, /*size=*/1, /*is_fp=*/0, /*LDR*/1,
                                 kSwImm12, Xbase, Wd_sw);

                // Patch TOP if pops occurred before this FSTSW.
                int16_t td = n.inputs[2];  // top_delta snapshot
                if (td != 0) {
                    int Wd_adj = alloc_free_gpr(*result);
                    if (td < 0)
                        emit_add_imm(buf, 0, /*is_sub=*/1, 0, 0, -td, Wd_top, Wd_adj);
                    else
                        emit_add_imm(buf, 0, /*is_sub=*/0, 0, 0, td, Wd_top, Wd_adj);
                    emit_and_imm(buf, 0, Wd_adj, 0, 0, 2, Wd_adj);
                    // BFI Wd_sw, Wd_adj, #11, #3 — patch TOP field
                    emit_bitfield(buf, 0, /*BFM*/1, 0, /*immr=*/21, /*imms=*/2,
                                  Wd_adj, Wd_sw);
                    free_gpr(*result, Wd_adj);
                }

                // BFI W_ax, Wd_sw, #0, #16 — write status_word into x86 AX
                int W_ax = n.inputs[1];  // destination register index (usually 0 = W0)
                emit_bitfield(buf, 0, /*BFM*/1, 0, /*immr=*/0, /*imms=*/15,
                              Wd_sw, W_ax);
                free_gpr(*result, Wd_sw);
            }
            break;
        }

        }  // switch

        // Free FPRs whose last use was this node.
        fprs.free_dead_inputs(*result, n, i);
    }

    // ── Epilogue: update x87 state ──────────────────────────────────────────

    // 1. Update TOP register.
    if (ctx.top_delta != 0) {
        if (ctx.top_delta < 0) {
            emit_add_imm(buf, 0, /*is_sub=*/1, 0, 0, -ctx.top_delta, Wd_top, Wd_top);
        } else {
            emit_add_imm(buf, 0, /*is_sub=*/0, 0, 0, ctx.top_delta, Wd_top, Wd_top);
        }
        emit_and_imm(buf, 0, Wd_top, 0, 0, 2, Wd_top);
    }

    // 2. Store modified stack values.
    // At this point, Wd_top holds the FINAL top. slot_val[d] tells us what value
    // should be at logical depth d relative to the final TOP.
    for (int d = 0; d < 8; d++) {
        int16_t val = ctx.slot_val[d];
        if (val < 0) continue;  // initial slot, unchanged (no store needed)
        // Skip redundant write-back: if the value is a ReadSt loaded from the
        // same physical slot it would be stored to, the store is a no-op.
        if (ctx.nodes[val].op == Op::ReadSt &&
            ctx.nodes[val].initial_depth == d + ctx.top_delta)
            continue;
        int Dd = fprs.get(val);
        if (Dd < 0) continue;   // dead or already freed
        emit_store_st(buf, Xbase, Wd_top, d, Wd_tmp, Dd, Xst_base);
    }

    // 3. Write TOP to status_word (if changed).
    if (ctx.top_delta != 0) {
        emit_store_top(buf, Xbase, Wd_top, Wd_tmp);
    }

    // 4. Update tag word for net pushes/pops.
    if (ctx.top_delta != 0) {
        int Wd_tmp2 = alloc_free_gpr(*result);
        if (ctx.top_delta > 0) {
            // Net pops: use the batch helper.
            int Wd_tagw = alloc_free_gpr(*result);
            emit_x87_tag_set_empty_batch(buf, Xbase, Wd_top, Wd_tmp, Wd_tmp2,
                                          Wd_tagw, ctx.top_delta);
            free_gpr(*result, Wd_tagw);
        } else {
            // Net pushes: clear tag bits for new slots.
            int abs_delta = -ctx.top_delta;
            int Wd_tagw = alloc_free_gpr(*result);
            emit_x87_tag_set_valid_batch(buf, Xbase, Wd_top, Wd_tmp, Wd_tmp2,
                                          Wd_tagw, abs_delta);
            free_gpr(*result, Wd_tagw);
        }
        free_gpr(*result, Wd_tmp2);
    }

    // 5. Free all remaining FPRs held by node values.
    for (int i = 0; i < ctx.num_nodes; i++) {
        if (fprs.node_fpr[i] >= 0) {
            free_fpr(*result, fprs.node_fpr[i]);
            fprs.node_fpr[i] = -1;
        }
    }

    // 6. Clean up cache deferred state (we handled everything inline).
    result->x87_cache.top_dirty = 0;
    result->x87_cache.tag_push_pending = 0;
    result->x87_cache.deferred_pop_count = 0;
    result->x87_cache.reset_perm();

    // 7. Free scratch GPR.
    free_gpr(*result, Wd_tmp);

    // 8. If cache is about to expire (run_remaining will hit 0 after ticks),
    //    free the cache GPRs. Otherwise, leave them pinned for the next run.
    if (result->x87_cache.run_remaining <= ctx.consumed) {
        // Cache will be deactivated by tick(). Release GPRs now so they don't
        // stay allocated past the run.
        // (tick() resets gprs_valid=0 but doesn't free the mask bits.)
        // The caller (Translator.cpp) resets free_gpr_mask after ticking.
    }
}

// ── Entry point ─────────────────────────────────────────────────────────────

int compile_run(TranslationResult* result, IRInstr* instr_array, int64_t num_instrs,
                int64_t start_idx, int run_length) {
    Context ctx;

    if (!build(ctx, instr_array, num_instrs, start_idx, run_length))
        return 0;

    optimize(ctx);
    lower(ctx, result);

    return ctx.consumed;
}

}  // namespace X87IR
