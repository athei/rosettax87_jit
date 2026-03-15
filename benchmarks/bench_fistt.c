/*
 * bench_fistt.c -- Benchmarks for FISTT (FISTTP) direct translation.
 * Covers: FISTT m16, FISTT m32, FISTT m64, plus FISTP m32 for comparison.
 */
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define TIMES 1000000
#define RUNS  5

static clock_t bench_fistt_m16(void) {
    clock_t start = clock();
    volatile int16_t r;
    for (int i = 0; i < TIMES; i++)
        __asm__ volatile (
            "fld1\n\t fld1\n\t faddp\n\t"  /* ST(0)=2 */
            "fisttps %0\n"
            : "=m"(r));
    return clock() - start;
}

static clock_t bench_fistt_m32(void) {
    clock_t start = clock();
    volatile int32_t r;
    for (int i = 0; i < TIMES; i++)
        __asm__ volatile (
            "fld1\n\t fld1\n\t faddp\n\t fld1\n\t faddp\n\t"  /* ST(0)=3 */
            "fisttpl %0\n"
            : "=m"(r));
    return clock() - start;
}

static clock_t bench_fistt_m64(void) {
    clock_t start = clock();
    volatile int64_t r;
    for (int i = 0; i < TIMES; i++)
        __asm__ volatile (
            "fld1\n\t fld1\n\t faddp\n\t fld1\n\t faddp\n\t"  /* ST(0)=3 */
            "fisttpll %0\n"
            : "=m"(r));
    return clock() - start;
}

/* FISTP m32 for comparison — same workload but with RC dispatch overhead */
static clock_t bench_fistp_m32(void) {
    clock_t start = clock();
    volatile int32_t r;
    for (int i = 0; i < TIMES; i++)
        __asm__ volatile (
            "fld1\n\t fld1\n\t faddp\n\t fld1\n\t faddp\n\t"  /* ST(0)=3 */
            "fistpl %0\n"
            : "=m"(r));
    return clock() - start;
}

int main(void) {
    struct { const char *name; clock_t (*fn)(void); } benches[] = {
        {"fistt_m16",  bench_fistt_m16},
        {"fistt_m32",  bench_fistt_m32},
        {"fistt_m64",  bench_fistt_m64},
        {"fistp_m32",  bench_fistp_m32},
    };
    int n = (int)(sizeof(benches) / sizeof(benches[0]));
    for (int i = 0; i < n; i++) {
        clock_t sum = 0;
        for (int r = 0; r < RUNS; r++) sum += benches[i].fn();
        printf("BENCH %s %lu\n", benches[i].name, (unsigned long)(sum / RUNS));
    }
    return 0;
}
