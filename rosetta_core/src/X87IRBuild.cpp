#include "rosetta_core/X87IR.h"

#include "rosetta_core/IRInstr.h"
#include "rosetta_core/IROperand.h"
#include "rosetta_core/Opcode.h"

namespace X87IR {

// ── Helpers ─────────────────────────────────────────────────────────────────

// Build a memory load node. Returns node ID or -1 on bail (e.g. m80).
static int16_t build_fp_load(Context& ctx, IROperand* op) {
    Op load_op;
    switch (op->mem.size) {
        case IROperandSize::S32: load_op = Op::LoadF32; break;
        case IROperandSize::S64: load_op = Op::LoadF64; break;
        default: return -1;  // m80 → bail
    }
    auto id = ctx.add_node(load_op);
    if (id < 0) return -1;
    ctx.nodes[id].mem_operand = op;
    return id;
}

static int16_t build_int_load(Context& ctx, IROperand* op) {
    Op load_op;
    switch (op->mem.size) {
        case IROperandSize::S16: load_op = Op::LoadI16; break;
        case IROperandSize::S32: load_op = Op::LoadI32; break;
        case IROperandSize::S64: load_op = Op::LoadI64; break;
        default: return -1;
    }
    auto id = ctx.add_node(load_op);
    if (id < 0) return -1;
    ctx.nodes[id].mem_operand = op;
    return id;
}

// Build a constant-push (FLDL2E, FLDL2T, etc.)
static int16_t build_const(Context& ctx, uint64_t bits) {
    auto id = ctx.add_node(Op::ConstF64);
    if (id < 0) return -1;
    ctx.nodes[id].imm_bits = bits;
    return id;
}

// Emit a memory store side-effect node.
static bool build_fp_store(Context& ctx, IROperand* op, int16_t val_node) {
    Op store_op;
    switch (op->mem.size) {
        case IROperandSize::S32: store_op = Op::StoreF32; break;
        case IROperandSize::S64: store_op = Op::StoreF64; break;
        default: return false;  // m80 → bail
    }
    auto id = ctx.add_node(store_op, val_node);
    if (id < 0) return false;
    ctx.nodes[id].mem_operand = op;
    return true;
}

// Emit an integer store side-effect node (FISTP/FIST/FISTTP).
static bool build_int_store(Context& ctx, IROperand* op, int16_t val_node, bool truncate) {
    Op store_op;
    switch (op->mem.size) {
        case IROperandSize::S16: store_op = Op::StoreI16; break;
        case IROperandSize::S32: store_op = Op::StoreI32; break;
        case IROperandSize::S64: store_op = Op::StoreI64; break;
        default: return false;
    }
    auto id = ctx.add_node(store_op, val_node);
    if (id < 0) return false;
    ctx.nodes[id].mem_operand = op;
    if (truncate) ctx.nodes[id].flags |= kTruncate;
    return true;
}

// ── Non-popping binary arithmetic ───────────────────────────────────────────

// Handles FADD/FSUB/FSUBR/FMUL/FDIV/FDIVR (non-popping).
// `base_op` is the IR op (FAdd/FSub/FMul/FDiv).
// `reversed` flips operand order (for FSUBR/FDIVR).
static bool build_arith(Context& ctx, IRInstr* instr, Op base_op, bool reversed) {
    if (instr->operands[0].kind == IROperandKind::Register) {
        // ST-ST path
        int depth_dst = instr->operands[0].reg.reg.index();
        int depth_src = instr->operands[1].reg.reg.index();
        auto dst_val = ctx.resolve(depth_dst);
        auto src_val = ctx.resolve(depth_src);
        if (dst_val < 0 || src_val < 0) return false;
        auto id = reversed
            ? ctx.add_node(base_op, src_val, dst_val)
            : ctx.add_node(base_op, dst_val, src_val);
        if (id < 0) return false;
        ctx.slot_val[depth_dst] = id;
    } else {
        // Memory path: ST(0) op= mem (or ST(0) = mem op ST(0) for reversed)
        auto st0 = ctx.resolve(0);
        if (st0 < 0) return false;
        auto mem_val = build_fp_load(ctx, &instr->operands[0]);
        if (mem_val < 0) return false;
        auto id = reversed
            ? ctx.add_node(base_op, mem_val, st0)
            : ctx.add_node(base_op, st0, mem_val);
        if (id < 0) return false;
        ctx.slot_val[0] = id;
    }
    return true;
}

// Handles FADDP/FSUBP/FSUBRP/FMULP/FDIVP/FDIVRP (popping).
static bool build_arithp(Context& ctx, IRInstr* instr, Op base_op, bool reversed) {
    int depth_dst = instr->operands[0].reg.reg.index();
    auto dst_val = ctx.resolve(depth_dst);
    auto src_val = ctx.resolve(0);
    if (dst_val < 0 || src_val < 0) return false;
    auto id = reversed
        ? ctx.add_node(base_op, src_val, dst_val)
        : ctx.add_node(base_op, dst_val, src_val);
    if (id < 0) return false;
    ctx.slot_val[depth_dst] = id;
    ctx.pop();
    return true;
}

// ── FCOM / FUCOM family ─────────────────────────────────────────────────────

static bool build_fcom(Context& ctx, IRInstr* instr, int num_pops, bool is_fcompp) {
    int16_t lhs = ctx.resolve(0);
    if (lhs < 0) return false;

    int16_t rhs;
    if (is_fcompp) {
        // FCOMPP/FUCOMPP: comparand is always ST(1).
        rhs = ctx.resolve(1);
    } else if (instr->operands[1].kind == IROperandKind::Register) {
        // FCOM/FCOMP ST(i): Rosetta encodes as [ST(0), ST(i)].
        // Comparand depth is in operands[1].
        int src_depth = instr->operands[1].reg.reg.index();
        rhs = ctx.resolve(src_depth);
    } else {
        // FCOM/FCOMP m32fp / m64fp: memory operand is operands[1].
        rhs = build_fp_load(ctx, &instr->operands[1]);
    }
    if (rhs < 0) return false;

    auto id = ctx.add_node(Op::FCmp, lhs, rhs);
    if (id < 0) return false;
    ctx.last_fcmp = id;

    for (int i = 0; i < num_pops; i++) ctx.pop();
    return true;
}

// ── Main build loop ─────────────────────────────────────────────────────────

bool build(Context& ctx, IRInstr* instr_array, int64_t num_instrs, int64_t start_idx,
           int run_length) {
    ctx.init();

    // FPR budget: we have 8 scratch FPRs (D24-D31). Reserve 2 for temporaries
    // used by the lowering (e.g. StoreF32 narrowing, address loads). That leaves
    // 6 for simultaneously live IR values. If the symbolic stack holds more than
    // 6 distinct computed values, bail — the lowering would exhaust the FPR pool.
    static constexpr int kMaxLiveValues = 14;

    for (int i = 0; i < run_length && (start_idx + i) < num_instrs; i++) {
        // Count distinct computed values in the symbolic stack.
        int live = 0;
        for (int d = 0; d < 8; d++) {
            if (ctx.slot_val[d] >= 0) live++;
        }
        if (live > kMaxLiveValues) break;

        auto* instr = &instr_array[start_idx + i];
        const auto op = instr->opcode;
        bool ok = true;

        switch (op) {
            // ── Loads / pushes ──────────────────────────────────────────
            case kOpcodeName_fldz: {
                auto id = ctx.add_node(Op::ConstZero);
                if (id < 0) { ok = false; break; }
                ctx.push(id);
                break;
            }
            case kOpcodeName_fld1: {
                auto id = ctx.add_node(Op::ConstOne);
                if (id < 0) { ok = false; break; }
                ctx.push(id);
                break;
            }
            case kOpcodeName_fldl2e: {
                auto id = build_const(ctx, 0x3FF71547652B82FEULL);
                if (id < 0) { ok = false; break; }
                ctx.push(id);
                break;
            }
            case kOpcodeName_fldl2t: {
                auto id = build_const(ctx, 0x400A934F0979A371ULL);
                if (id < 0) { ok = false; break; }
                ctx.push(id);
                break;
            }
            case kOpcodeName_fldlg2: {
                auto id = build_const(ctx, 0x3FD34413509F79FFULL);
                if (id < 0) { ok = false; break; }
                ctx.push(id);
                break;
            }
            case kOpcodeName_fldln2: {
                auto id = build_const(ctx, 0x3FE62E42FEFA39EFULL);
                if (id < 0) { ok = false; break; }
                ctx.push(id);
                break;
            }
            case kOpcodeName_fldpi: {
                auto id = build_const(ctx, 0x400921FB54442D18ULL);
                if (id < 0) { ok = false; break; }
                ctx.push(id);
                break;
            }
            case kOpcodeName_fld: {
                if (instr->operands[0].kind == IROperandKind::Register) {
                    int depth = instr->operands[1].reg.reg.index();
                    auto val = ctx.resolve(depth);
                    if (val < 0) { ok = false; break; }
                    ctx.push(val);
                } else {
                    auto val = build_fp_load(ctx, &instr->operands[0]);
                    if (val < 0) { ok = false; break; }
                    ctx.push(val);
                }
                break;
            }
            case kOpcodeName_fild: {
                auto val = build_int_load(ctx, &instr->operands[0]);
                if (val < 0) { ok = false; break; }
                ctx.push(val);
                break;
            }

            // ── Stores ──────────────────────────────────────────────────
            case kOpcodeName_fst:
            case kOpcodeName_fst_stack:
            case kOpcodeName_fstp:
            case kOpcodeName_fstp_stack: {
                bool is_pop = (op == kOpcodeName_fstp || op == kOpcodeName_fstp_stack);
                auto st0 = ctx.resolve(0);
                if (st0 < 0) { ok = false; break; }

                if (instr->operands[0].kind == IROperandKind::Register) {
                    // Register path: copy ST(0) → ST(i)
                    int dst_depth = instr->operands[0].reg.reg.index();
                    if (dst_depth != 0)
                        ctx.slot_val[dst_depth] = st0;
                } else {
                    // Memory path
                    if (!build_fp_store(ctx, &instr->operands[0], st0)) {
                        ok = false; break;
                    }
                }
                if (is_pop) ctx.pop();
                break;
            }

            // ── Integer stores ──────────────────────────────────────────
            case kOpcodeName_fistp:
            case kOpcodeName_fist:
            case kOpcodeName_fisttp: {
                auto st0 = ctx.resolve(0);
                if (st0 < 0) { ok = false; break; }
                if (!build_int_store(ctx, &instr->operands[0], st0,
                                     op == kOpcodeName_fisttp)) {
                    ok = false; break;
                }
                if (op != kOpcodeName_fist) ctx.pop();
                break;
            }

            // ── Non-popping arithmetic ──────────────────────────────────
            case kOpcodeName_fadd:
                ok = build_arith(ctx, instr, Op::FAdd, false);
                break;
            case kOpcodeName_fsub:
                ok = build_arith(ctx, instr, Op::FSub, false);
                break;
            case kOpcodeName_fsubr:
                ok = build_arith(ctx, instr, Op::FSub, true);
                break;
            case kOpcodeName_fmul:
                ok = build_arith(ctx, instr, Op::FMul, false);
                break;
            case kOpcodeName_fdiv:
                ok = build_arith(ctx, instr, Op::FDiv, false);
                break;
            case kOpcodeName_fdivr:
                ok = build_arith(ctx, instr, Op::FDiv, true);
                break;

            // ── Integer arithmetic (memory-only, ST(0) op= mem) ────────
            case kOpcodeName_fiadd:
            case kOpcodeName_fisub:
            case kOpcodeName_fisubr:
            case kOpcodeName_fimul:
            case kOpcodeName_fidiv:
            case kOpcodeName_fidivr: {
                auto st0 = ctx.resolve(0);
                if (st0 < 0) { ok = false; break; }
                auto mem_val = build_int_load(ctx, &instr->operands[0]);
                if (mem_val < 0) { ok = false; break; }
                Op arith_op;
                bool reversed;
                switch (op) {
                    case kOpcodeName_fiadd:  arith_op = Op::FAdd; reversed = false; break;
                    case kOpcodeName_fisub:  arith_op = Op::FSub; reversed = false; break;
                    case kOpcodeName_fisubr: arith_op = Op::FSub; reversed = true;  break;
                    case kOpcodeName_fimul:  arith_op = Op::FMul; reversed = false; break;
                    case kOpcodeName_fidiv:  arith_op = Op::FDiv; reversed = false; break;
                    case kOpcodeName_fidivr: arith_op = Op::FDiv; reversed = true;  break;
                    default: __builtin_unreachable();
                }
                auto id = reversed
                    ? ctx.add_node(arith_op, mem_val, st0)
                    : ctx.add_node(arith_op, st0, mem_val);
                if (id < 0) { ok = false; break; }
                ctx.slot_val[0] = id;
                break;
            }

            // ── Popping arithmetic ──────────────────────────────────────
            case kOpcodeName_faddp:
                ok = build_arithp(ctx, instr, Op::FAdd, false);
                break;
            case kOpcodeName_fsubp:
                ok = build_arithp(ctx, instr, Op::FSub, false);
                break;
            case kOpcodeName_fsubrp:
                ok = build_arithp(ctx, instr, Op::FSub, true);
                break;
            case kOpcodeName_fmulp:
                ok = build_arithp(ctx, instr, Op::FMul, false);
                break;
            case kOpcodeName_fdivp:
                ok = build_arithp(ctx, instr, Op::FDiv, false);
                break;
            case kOpcodeName_fdivrp:
                ok = build_arithp(ctx, instr, Op::FDiv, true);
                break;

            // ── Unary ───────────────────────────────────────────────────
            case kOpcodeName_fchs: {
                auto st0 = ctx.resolve(0);
                if (st0 < 0) { ok = false; break; }
                auto id = ctx.add_node(Op::FNeg, st0);
                if (id < 0) { ok = false; break; }
                ctx.slot_val[0] = id;
                break;
            }
            case kOpcodeName_fabs: {
                auto st0 = ctx.resolve(0);
                if (st0 < 0) { ok = false; break; }
                auto id = ctx.add_node(Op::FAbs, st0);
                if (id < 0) { ok = false; break; }
                ctx.slot_val[0] = id;
                break;
            }
            case kOpcodeName_fsqrt: {
                auto st0 = ctx.resolve(0);
                if (st0 < 0) { ok = false; break; }
                auto id = ctx.add_node(Op::FSqrt, st0);
                if (id < 0) { ok = false; break; }
                ctx.slot_val[0] = id;
                break;
            }

            // ── FXCH — compile-time swap ────────────────────────────────
            case kOpcodeName_fxch: {
                // Rosetta encodes FXCH as [ST(0), ST(i)] — target depth is operands[1].
                int depth = instr->operands[1].reg.reg.index();
                if (depth <= 0 || depth >= 8) { ok = false; break; }
                // Force-resolve both slots so the epilogue can track the swap.
                // Without this, two unresolved initial slots (val < 0) would be
                // swapped symbolically but the epilogue would skip both stores.
                ctx.resolve(0);
                ctx.resolve(depth);
                auto tmp = ctx.slot_val[0];
                ctx.slot_val[0] = ctx.slot_val[depth];
                ctx.slot_val[depth] = tmp;
                break;
            }

            // ── Compare ─────────────────────────────────────────────────
            case kOpcodeName_fcom:
            case kOpcodeName_fucom:
                ok = build_fcom(ctx, instr, 0, false);
                break;
            case kOpcodeName_fcomp:
            case kOpcodeName_fucomp:
                ok = build_fcom(ctx, instr, 1, false);
                break;
            case kOpcodeName_fcompp:
            case kOpcodeName_fucompp:
                ok = build_fcom(ctx, instr, 2, true);
                break;

            case kOpcodeName_ftst: {
                auto st0 = ctx.resolve(0);
                if (st0 < 0) { ok = false; break; }
                auto id = ctx.add_node(Op::FTst, st0);
                if (id < 0) { ok = false; break; }
                ctx.last_fcmp = id;
                break;
            }

            // ── FSTSW AX ───────────────────────────────────────────────
            case kOpcodeName_fstsw: {
                // Bail on memory form (extremely rare); only handle FSTSW AX.
                if (instr->operands[0].kind != IROperandKind::Register) {
                    ok = false; break;
                }
                auto id = ctx.add_node(Op::FStsw);
                if (id < 0) { ok = false; break; }
                // inputs[0] = prior FCmp/FTst node (for fusion detection)
                ctx.nodes[id].inputs[0] = ctx.last_fcmp;
                // inputs[1] = destination register index (0 = AX → W0)
                ctx.nodes[id].inputs[1] = static_cast<int16_t>(
                    instr->operands[0].reg.reg.index());
                // inputs[2] = top_delta snapshot (for TOP patching in lowering)
                ctx.nodes[id].inputs[2] = ctx.top_delta;
                break;
            }

            // ── Bail on everything else ─────────────────────────────────
            default:
                ok = false;
                break;
        }

        if (!ok) break;
        ctx.consumed = static_cast<int16_t>(i + 1);
    }

    return ctx.consumed >= 2;  // only worthwhile if we consumed 2+ instructions
}

}  // namespace X87IR
