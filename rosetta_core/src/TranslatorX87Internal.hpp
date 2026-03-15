#pragma once

// Internal helpers shared between TranslatorX87.cpp and TranslatorX87Fusion.cpp.
// These are inline so both translation units get their own copy without ODR issues.

#include <utility>

#include "rosetta_core/TranslationResult.h"
#include "rosetta_core/TranslatorHelpers.hpp"
#include "rosetta_core/TranslatorX87Helpers.hpp"

namespace TranslatorX87 {

// ── OPT-G: Deferred FXCH — compile-time depth resolution ────────────────────
//
// When the permutation map is active (perm_dirty), resolve_depth remaps a
// logical stack depth to the physical depth that should be passed to
// emit_load_st / emit_store_st.  This happens entirely at JIT-compile time
// (no runtime cost).

inline int resolve_depth(const TranslationResult& a1, int logical_depth) {
    if (!a1.x87_cache.perm_dirty) return logical_depth;
    return a1.x87_cache.perm[logical_depth];
}

// ── Preamble / epilogue used by every translate_* and try_fuse_* function ────

inline auto x87_begin(TranslationResult& a1, AssemblerBuffer& buf) -> std::pair<int, int> {
    if (a1.x87_cache.run_remaining > 0 && a1.x87_cache.gprs_valid) {
        return {a1.x87_cache.base_gpr, a1.x87_cache.top_gpr};
    }

    const int Xbase = alloc_gpr(a1, 0);
    const int Wd_top = alloc_gpr(a1, 1);
    emit_x87_base(buf, a1, Xbase);
    emit_load_top(buf, a1, Xbase, Wd_top);

    if (a1.x87_cache.run_remaining > 0) {
        a1.x87_cache.base_gpr = static_cast<int8_t>(Xbase);
        a1.x87_cache.top_gpr = static_cast<int8_t>(Wd_top);

        const int Xst_base = alloc_gpr(a1, 6);
        emit_add_imm(buf, /*is_64bit=*/1, /*is_sub=*/0, /*is_set_flags=*/0,
                     /*shift=*/0, kX87RegFileOff, Xbase, Xst_base);
        a1.x87_cache.st_base_gpr = static_cast<int8_t>(Xst_base);
        a1.x87_cache.gprs_valid = 1;
    }

    return {Xbase, Wd_top};
}

inline int x87_get_st_base(TranslationResult& a1) {
    return a1.x87_cache.gprs_valid ? a1.x87_cache.st_base_gpr : -1;
}

// OPT-G: Flush deferred permutation before push/pop.
// Push/pop change TOP, which invalidates the perm map (it maps depths relative
// to TOP). Rather than trying to adjust the map, flush it and reset to identity.
inline void perm_flush_before_stack_change(AssemblerBuffer& buf, TranslationResult& a1,
                                            int Xbase, int Wd_top, int Wd_tmp) {
    if (a1.x87_cache.perm_dirty) {
        const int Xst_base = x87_get_st_base(a1);
        const int Dd_save = alloc_free_fpr(a1);
        const int Dd_chain = alloc_free_fpr(a1);
        emit_x87_perm_flush(buf, Xbase, Wd_top, Wd_tmp, a1.x87_cache.perm,
                            Xst_base, Dd_save, Dd_chain);
        free_fpr(a1, Dd_chain);
        free_fpr(a1, Dd_save);
        a1.x87_cache.reset_perm();
    }
}

inline void x87_end(TranslationResult& a1, AssemblerBuffer& buf, int Xbase, int Wd_top,
                    int Wd_tmp, int consumed = 1) {
    // OPT-G: flush deferred permutation at end of run.
    if (a1.x87_cache.perm_dirty && a1.x87_cache.run_remaining <= consumed) {
        const int Xst_base = x87_get_st_base(a1);
        const int Dd_save = alloc_free_fpr(a1);
        const int Dd_chain = alloc_free_fpr(a1);
        emit_x87_perm_flush(buf, Xbase, Wd_top, Wd_tmp, a1.x87_cache.perm,
                            Xst_base, Dd_save, Dd_chain);
        free_fpr(a1, Dd_chain);
        free_fpr(a1, Dd_save);
        a1.x87_cache.reset_perm();
    }

    // OPT-D: flush any deferred tag-valid update at end of run.
    // For multi-instruction fusions (consumed > 1), tick() will decrement
    // run_remaining by `consumed` after this call.  We must flush before
    // that silent clear fires.
    if (a1.x87_cache.tag_push_pending && a1.x87_cache.run_remaining <= consumed) {
        const int Wd_tmp2 = alloc_free_gpr(a1);
        emit_x87_tag_clear(buf, Xbase, Wd_top, Wd_tmp, Wd_tmp2);
        free_gpr(a1, Wd_tmp2);
        a1.x87_cache.tag_push_pending = 0;
    }

    if (a1.x87_cache.top_dirty && a1.x87_cache.run_remaining <= consumed) {
        emit_store_top(buf, Xbase, Wd_top, Wd_tmp);
        a1.x87_cache.top_dirty = 0;
    }

    if (a1.x87_cache.run_remaining > 0) {
        return;
    }
    free_gpr(a1, Wd_top);
    free_gpr(a1, Xbase);
}

inline void x87_cache_force_release(TranslationResult& a1, AssemblerBuffer& buf) {
    // OPT-G: flush deferred permutation before releasing.
    if (a1.x87_cache.perm_dirty && a1.x87_cache.gprs_valid) {
        const int Xst_base = x87_get_st_base(a1);
        const int tmp = alloc_gpr(a1, 2);
        const int Dd_save = alloc_free_fpr(a1);
        const int Dd_chain = alloc_free_fpr(a1);
        emit_x87_perm_flush(buf, a1.x87_cache.base_gpr, a1.x87_cache.top_gpr,
                            tmp, a1.x87_cache.perm, Xst_base, Dd_save, Dd_chain);
        free_fpr(a1, Dd_chain);
        free_fpr(a1, Dd_save);
        free_gpr(a1, tmp);
        a1.x87_cache.reset_perm();
    }
    // OPT-D: flush deferred tag-valid update before releasing.
    if (a1.x87_cache.tag_push_pending && a1.x87_cache.gprs_valid) {
        const int tmp = alloc_gpr(a1, 2);
        const int tmp2 = alloc_gpr(a1, 3);
        emit_x87_tag_clear(buf, a1.x87_cache.base_gpr, a1.x87_cache.top_gpr, tmp, tmp2);
        free_gpr(a1, tmp2);
        free_gpr(a1, tmp);
        a1.x87_cache.tag_push_pending = 0;
    }
    if (a1.x87_cache.top_dirty && a1.x87_cache.gprs_valid) {
        const int tmp = alloc_gpr(a1, 2);
        emit_store_top(buf, a1.x87_cache.base_gpr, a1.x87_cache.top_gpr, tmp);
        free_gpr(a1, tmp);
        a1.x87_cache.top_dirty = 0;
    }
    if (a1.x87_cache.gprs_valid) {
        a1.free_gpr_mask |= (1u << a1.x87_cache.base_gpr);
        a1.free_gpr_mask |= (1u << a1.x87_cache.top_gpr);
        a1.free_gpr_mask |= (1u << a1.x87_cache.st_base_gpr);
    }
    a1.x87_cache.invalidate();
}

// ── OPT-C/D: Flush helpers ──────────────────────────────────────────────────

inline void x87_flush_top(AssemblerBuffer& buf, TranslationResult& a1, int Xbase, int Wd_top,
                          int Wd_tmp) {
    if (a1.x87_cache.top_dirty) {
        emit_store_top(buf, Xbase, Wd_top, Wd_tmp);
        a1.x87_cache.top_dirty = 0;
    }
}

// OPT-D: Flush deferred tag-valid update.  Must be called before any code that
// reads the tag word (e.g. FSTSW, FTST, FCOM conditions, FCMOV).
inline void x87_flush_tags(AssemblerBuffer& buf, TranslationResult& a1, int Xbase, int Wd_top,
                           int Wd_tmp, int Wd_tmp2) {
    if (a1.x87_cache.tag_push_pending) {
        emit_x87_tag_clear(buf, Xbase, Wd_top, Wd_tmp, Wd_tmp2);
        a1.x87_cache.tag_push_pending = 0;
    }
}

// ── OPT-C/D: Push/pop wrappers that manage deferred writeback + tag flags ────

inline void x87_push(AssemblerBuffer& buf, TranslationResult& a1, int Xbase, int Wd_top, int Wd_tmp,
                     int Wd_tmp2) {
    if (a1.x87_cache.run_remaining > 0) {
        // OPT-D: If there's already a pending tag from a prior push, flush it
        // before creating a new one.  Also flush store_top so that the
        // push-pop cancellation in x87_pop sees the correct memory TOP.
        if (a1.x87_cache.tag_push_pending) {
            emit_x87_tag_clear(buf, Xbase, Wd_top, Wd_tmp, Wd_tmp2);
            a1.x87_cache.tag_push_pending = 0;
        }
        if (a1.x87_cache.top_dirty) {
            emit_store_top(buf, Xbase, Wd_top, Wd_tmp);
            a1.x87_cache.top_dirty = 0;
        }
        perm_flush_before_stack_change(buf, a1, Xbase, Wd_top, Wd_tmp);
        emit_x87_push_fully_deferred(buf, Wd_top);
        a1.x87_cache.top_dirty = 1;
        a1.x87_cache.tag_push_pending = 1;
    } else {
        emit_x87_push(buf, Xbase, Wd_top, Wd_tmp, Wd_tmp2);
    }
}

inline void x87_pop(AssemblerBuffer& buf, TranslationResult& a1, int Xbase, int Wd_top,
                    int Wd_tmp) {
    if (a1.x87_cache.run_remaining > 0) {
        perm_flush_before_stack_change(buf, a1, Xbase, Wd_top, Wd_tmp);
        if (a1.x87_cache.tag_push_pending && a1.x87_cache.top_dirty) {
            // OPT-D: Full push-pop cancellation.  Both tag updates cancel, and
            // memory already has the correct final TOP (pre-push = post-pop).
            emit_x87_pop_top_only(buf, Wd_top);
            a1.x87_cache.tag_push_pending = 0;
            a1.x87_cache.top_dirty = 0;
        } else if (a1.x87_cache.tag_push_pending) {
            // Tag pending but TOP was flushed to memory between push and pop.
            // Tags still cancel, but need to store the new TOP value.
            emit_x87_pop_top_only(buf, Wd_top);
            emit_store_top(buf, Xbase, Wd_top, Wd_tmp);
            a1.x87_cache.tag_push_pending = 0;
            a1.x87_cache.top_dirty = 0;
        } else {
            const int Wd_tmp2 = alloc_free_gpr(a1);
            emit_x87_pop_deferred(buf, Xbase, Wd_top, Wd_tmp, Wd_tmp2);
            free_gpr(a1, Wd_tmp2);
            a1.x87_cache.top_dirty = 1;
        }
    } else {
        const int Wd_tmp2 = alloc_free_gpr(a1);
        emit_x87_pop(buf, Xbase, Wd_top, Wd_tmp, Wd_tmp2);
        free_gpr(a1, Wd_tmp2);
        a1.x87_cache.top_dirty = 0;
    }
}

inline void x87_pop_n(AssemblerBuffer& buf, TranslationResult& a1, int Xbase, int Wd_top,
                      int Wd_tmp, int n) {
    const int Wd_tmp2 = alloc_free_gpr(a1);
    // OPT-D: flush pending tag before multi-pop (cancellation is only 1:1).
    x87_flush_tags(buf, a1, Xbase, Wd_top, Wd_tmp, Wd_tmp2);
    if (a1.x87_cache.run_remaining > 0) {
        perm_flush_before_stack_change(buf, a1, Xbase, Wd_top, Wd_tmp);
        emit_x87_pop_n_deferred(buf, Xbase, Wd_top, Wd_tmp, Wd_tmp2, n);
        a1.x87_cache.top_dirty = 1;
    } else {
        emit_x87_pop_n(buf, Xbase, Wd_top, Wd_tmp, Wd_tmp2, n);
        a1.x87_cache.top_dirty = 0;
    }
    free_gpr(a1, Wd_tmp2);
}

}  // namespace TranslatorX87
