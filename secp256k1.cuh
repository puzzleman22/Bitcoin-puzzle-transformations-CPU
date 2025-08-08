#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define BIGINT_WORDS 8

struct BigInt {
    uint32_t data[BIGINT_WORDS];
};

struct ECPoint {
    BigInt x, y;
    bool infinity;
};

struct ECPointJac {
    BigInt X, Y, Z;
    bool infinity;
};

__constant__ BigInt const_p;
__constant__ ECPointJac const_G_jacobian;
__constant__ BigInt const_n;

#define WINDOW_SIZE 16
__device__ ECPointJac G_precomp[1 << WINDOW_SIZE];



__host__ __device__ __forceinline__ void init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

__host__ __device__ __forceinline__ void copy_bigint(BigInt *dest, const BigInt *src) {
	#pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}

__host__ __device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
	#pragma unroll
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

__host__ __device__ __forceinline__ bool is_zero(const BigInt *a) {
	#pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (a->data[i]) return false;
    }
    return true;
}

__host__ __device__ __forceinline__ int get_bit(const BigInt *a, int i) {
    int word_idx = i >> 5; // i / 32
    int bit_idx = i & 31;  // i % 32
    if (word_idx >= BIGINT_WORDS) return 0;
    return (a->data[word_idx] >> bit_idx) & 1;
}

__device__ __forceinline__ void ptx_u256Add(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile(
        "add.cc.u32 %0, %8, %16;\n\t"
        "addc.cc.u32 %1, %9, %17;\n\t"
        "addc.cc.u32 %2, %10, %18;\n\t"
        "addc.cc.u32 %3, %11, %19;\n\t"
        "addc.cc.u32 %4, %12, %20;\n\t"
        "addc.cc.u32 %5, %13, %21;\n\t"
        "addc.cc.u32 %6, %14, %22;\n\t"
        "addc.u32 %7, %15, %23;\n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

__device__ __forceinline__ void ptx_u256Sub(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile(
        "sub.cc.u32 %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.u32 %7, %15, %23;\n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

// Optimized multiply_bigint_by_const with unrolling
__device__ __forceinline__ void multiply_bigint_by_const(const BigInt *a, uint32_t c, uint32_t result[9]) {
    uint32_t carry = 0;
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint32_t lo, hi;
        asm volatile(
            "mul.lo.u32 %0, %2, %3;\n\t"
            "mul.hi.u32 %1, %2, %3;\n\t"
            "add.cc.u32 %0, %0, %4;\n\t"
            "addc.u32 %1, %1, 0;\n\t"
            : "=r"(lo), "=r"(hi)
            : "r"(a->data[i]), "r"(c), "r"(carry)
        );
        result[i] = lo;
        carry = hi;
    }
    result[8] = carry;
}

// Optimized shift_left_word
__device__ __forceinline__ void shift_left_word(const BigInt *a, uint32_t result[9]) {
    result[0] = 0;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i+1] = a->data[i];
    }
}

__device__ __forceinline__ void add_9word(uint32_t r[9], const uint32_t addend[9]) {
    // Use PTX add with carry chain for efficient 9-word addition
    asm volatile(
        "add.cc.u32 %0, %0, %9;\n\t"      // r[0] += addend[0], set carry
        "addc.cc.u32 %1, %1, %10;\n\t"    // r[1] += addend[1] + carry, set carry
        "addc.cc.u32 %2, %2, %11;\n\t"    // r[2] += addend[2] + carry, set carry
        "addc.cc.u32 %3, %3, %12;\n\t"    // r[3] += addend[3] + carry, set carry
        "addc.cc.u32 %4, %4, %13;\n\t"    // r[4] += addend[4] + carry, set carry
        "addc.cc.u32 %5, %5, %14;\n\t"    // r[5] += addend[5] + carry, set carry
        "addc.cc.u32 %6, %6, %15;\n\t"    // r[6] += addend[6] + carry, set carry
        "addc.cc.u32 %7, %7, %16;\n\t"    // r[7] += addend[7] + carry, set carry
        "addc.u32 %8, %8, %17;\n\t"       // r[8] += addend[8] + carry (no carry out needed)
        : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]), 
          "+r"(r[4]), "+r"(r[5]), "+r"(r[6]), "+r"(r[7]), 
          "+r"(r[8])
        : "r"(addend[0]), "r"(addend[1]), "r"(addend[2]), "r"(addend[3]),
          "r"(addend[4]), "r"(addend[5]), "r"(addend[6]), "r"(addend[7]),
          "r"(addend[8])
    );
}

__device__ __forceinline__ void convert_9word_to_bigint(const uint32_t r[9], BigInt *res) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = r[i];
    }
}

__device__ __forceinline__ void mul_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    // Keep EXACT multiplication code that works
    uint64_t prod[16] = {0};
    
    // Optimize: Use shared memory for better cache locality if available
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t carry = 0;
        uint32_t ai = a->data[i];  // Cache in register
        
        #pragma unroll
        for (int j = 0; j < BIGINT_WORDS; j++) {
            uint32_t lo, hi;
            asm volatile(
                "mul.lo.u32 %0, %2, %3;\n\t"
                "mul.hi.u32 %1, %2, %3;\n\t"
                : "=r"(lo), "=r"(hi)
                : "r"(ai), "r"(b->data[j])  // Use cached value
            );
            uint64_t mul = ((uint64_t)hi << 32) | lo;
            uint64_t sum = prod[i + j] + mul + carry;
            prod[i + j] = (uint32_t)sum;
            carry = sum >> 32;
        }
        prod[i + BIGINT_WORDS] += carry;
    }
    
    // Convert to 32-bit array - keep exactly the same but unroll
    uint32_t prod32[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        prod32[i] = (uint32_t)(prod[i] & 0xFFFFFFFFULL);
    }
    
    // Optimize: Combine L and H extraction with Rext initialization
    uint32_t Rext[9];
    BigInt H;
    
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        Rext[i] = prod32[i];  // L part goes directly to Rext
        H.data[i] = prod32[i + BIGINT_WORDS];  // H part
    }
    Rext[8] = 0;
    
    // Optimize multiply_bigint_by_const with better PTX usage
    uint32_t H977[9];
    {
        uint32_t carry = 0;
        #pragma unroll
        for (int i = 0; i < BIGINT_WORDS; i++) {
            uint32_t lo, hi;
            asm volatile(
                "mad.lo.cc.u32 %0, %2, %3, %4;\n\t"  // lo = a*977 + carry (with carry out)
                "madc.hi.u32 %1, %2, %3, 0;\n\t"     // hi = high(a*977) + carry in
                : "=r"(lo), "=r"(hi)
                : "r"(H.data[i]), "r"(977), "r"(carry)
            );
            H977[i] = lo;
            carry = hi;
        }
        H977[8] = carry;
    }
    
    // Use optimized add_9word
    add_9word(Rext, H977);
    
    uint32_t Hshift[9] = {0};
    shift_left_word(&H, Hshift);
    add_9word(Rext, Hshift);
    
    // Keep overflow handling exactly the same
    if (Rext[8]) {
        uint32_t extra[9] = {0};
        BigInt extraBI;
        init_bigint(&extraBI, Rext[8]);
        Rext[8] = 0;
        
        uint32_t extra977[9] = {0}, extraShift[9] = {0};
        
        // Optimize: inline small multiply for single word
        {
            uint32_t lo, hi;
            asm volatile(
                "mul.lo.u32 %0, %2, %3;\n\t"
                "mul.hi.u32 %1, %2, %3;\n\t"
                : "=r"(lo), "=r"(hi)
                : "r"(extraBI.data[0]), "r"(977)
            );
            extra977[0] = lo;
            extra977[1] = hi;
        }
        
        shift_left_word(&extraBI, extraShift);
        
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            extra[i] = extra977[i];
        }
        add_9word(extra, extraShift);
        add_9word(Rext, extra);
    }
    
    // Final reduction - exactly the same
    BigInt R_temp;
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) {
        R_temp.data[i] = Rext[i];
    }
    
    if (Rext[8] || compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }
    if (compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }
    
    copy_bigint(res, &R_temp);
}
__device__ __forceinline__ void sub_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt temp;
    if (compare_bigint(a, b) < 0) {
         BigInt sum;
         ptx_u256Add(&sum, a, &const_p);
         ptx_u256Sub(&temp, &sum, b);
    } else {
         ptx_u256Sub(&temp, a, b);
    }
    copy_bigint(res, &temp);
}

__device__ __forceinline__ void scalar_mod_n(BigInt *res, const BigInt *a) {
    if (compare_bigint(a, &const_n) >= 0) {
        ptx_u256Sub(res, a, &const_n);
    } else {
        copy_bigint(res, a);
    }
}

__device__ __forceinline__ void add_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t carry;
    
    // Use PTX for addition with carry flag
    asm volatile(
        "add.cc.u32 %0, %9, %17;\n\t"
        "addc.cc.u32 %1, %10, %18;\n\t"
        "addc.cc.u32 %2, %11, %19;\n\t"
        "addc.cc.u32 %3, %12, %20;\n\t"
        "addc.cc.u32 %4, %13, %21;\n\t"
        "addc.cc.u32 %5, %14, %22;\n\t"
        "addc.cc.u32 %6, %15, %23;\n\t"
        "addc.cc.u32 %7, %16, %24;\n\t"
        "addc.u32 %8, 0, 0;\n\t"  // capture final carry
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7]),
          "=r"(carry)
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
    
    if (carry || compare_bigint(res, &const_p) >= 0) {
        ptx_u256Sub(res, res, &const_p);
    }
}

__device__ void modexp(BigInt *res, const BigInt *base, const BigInt *exp) {
    BigInt result;
    init_bigint(&result, 1);
    BigInt b;
    copy_bigint(&b, base);
    for (int i = 0; i < 256; i++) {
         if (get_bit(exp, i)) {
              mul_mod_device(&result, &result, &b);
         }
         mul_mod_device(&b, &b, &b);
    }
    copy_bigint(res, &result);
}

__device__ void mod_inverse(BigInt *res, const BigInt *a) {
    BigInt p_minus_2, two;
    init_bigint(&two, 2);
    ptx_u256Sub(&p_minus_2, &const_p, &two);
    modexp(res, a, &p_minus_2);
}

__device__ __forceinline__ void point_set_infinity_jac(ECPointJac *P) {
    P->infinity = true;
}

__device__ __forceinline__ void point_copy_jac(ECPointJac *dest, const ECPointJac *src) {
    copy_bigint(&dest->X, &src->X);
    copy_bigint(&dest->Y, &src->Y);
    copy_bigint(&dest->Z, &src->Z);
    dest->infinity = src->infinity;
}

__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P) {
    if (P->infinity || is_zero(&P->Y)) {
        point_set_infinity_jac(R);
        return;
    }
    BigInt A, B, C, D, X3, Y3, Z3, temp, temp2;
    mul_mod_device(&A, &P->Y, &P->Y);
    mul_mod_device(&temp, &P->X, &A);
    init_bigint(&temp2, 4);
    mul_mod_device(&B, &temp, &temp2);
    mul_mod_device(&temp, &A, &A);
    init_bigint(&temp2, 8);
    mul_mod_device(&C, &temp, &temp2);
    mul_mod_device(&temp, &P->X, &P->X);
    init_bigint(&temp2, 3);
    mul_mod_device(&D, &temp, &temp2);
    BigInt D2, two, twoB;
    mul_mod_device(&D2, &D, &D);
    init_bigint(&two, 2);
    mul_mod_device(&twoB, &B, &two);
    sub_mod_device(&X3, &D2, &twoB);
    sub_mod_device(&temp, &B, &X3);
    mul_mod_device(&temp, &D, &temp);
    sub_mod_device(&Y3, &temp, &C);
    init_bigint(&temp, 2);
    mul_mod_device(&temp, &temp, &P->Y);
    mul_mod_device(&Z3, &temp, &P->Z);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q) {
    if (P->infinity) { point_copy_jac(R, Q); return; }
    if (Q->infinity) { point_copy_jac(R, P); return; }

    BigInt Z1Z1, Z2Z2, U1, U2, S1, S2, H, R_big, H2, H3, U1H2, X3, Y3, Z3, temp;
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    mul_mod_device(&Z2Z2, &Q->Z, &Q->Z);
    mul_mod_device(&U1, &P->X, &Z2Z2);
    mul_mod_device(&U2, &Q->X, &Z1Z1);
    BigInt Z2_cubed, Z1_cubed;
    mul_mod_device(&temp, &Z2Z2, &Q->Z); copy_bigint(&Z2_cubed, &temp);
    mul_mod_device(&temp, &Z1Z1, &P->Z); copy_bigint(&Z1_cubed, &temp);
    mul_mod_device(&S1, &P->Y, &Z2_cubed);
    mul_mod_device(&S2, &Q->Y, &Z1_cubed);

    if (compare_bigint(&U1, &U2) == 0) {
        if (compare_bigint(&S1, &S2) != 0) {
            point_set_infinity_jac(R);
            return;
        } else {
            double_point_jac(R, P);
            return;
        }
    }
    sub_mod_device(&H, &U2, &U1);
    sub_mod_device(&R_big, &S2, &S1);
    mul_mod_device(&H2, &H, &H);
    mul_mod_device(&H3, &H2, &H);
    mul_mod_device(&U1H2, &U1, &H2);
    BigInt R2, two, twoU1H2;
    mul_mod_device(&R2, &R_big, &R_big);
    init_bigint(&two, 2);
    mul_mod_device(&twoU1H2, &U1H2, &two);
    sub_mod_device(&temp, &R2, &H3);
    sub_mod_device(&X3, &temp, &twoU1H2);
    sub_mod_device(&temp, &U1H2, &X3);
    mul_mod_device(&temp, &R_big, &temp);
    mul_mod_device(&Y3, &S1, &H3);
    sub_mod_device(&Y3, &temp, &Y3);
    mul_mod_device(&temp, &P->Z, &Q->Z);
    mul_mod_device(&Z3, &temp, &H);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}


__device__ void jacobian_to_affine(ECPoint *R, const ECPointJac *P) {
    if (P->infinity) {
        R->infinity = true;
        init_bigint(&R->x, 0);
        init_bigint(&R->y, 0);
        return;
    }
    BigInt Zinv, Zinv2, Zinv3;
    mod_inverse(&Zinv, &P->Z);
    mul_mod_device(&Zinv2, &Zinv, &Zinv);
    mul_mod_device(&Zinv3, &Zinv2, &Zinv);
    mul_mod_device(&R->x, &P->X, &Zinv2);
    mul_mod_device(&R->y, &P->Y, &Zinv3);
    R->infinity = false;
}


__device__ void scalar_multiply_jac_device(ECPointJac *result, const ECPointJac *point, const BigInt *scalar) {

    const int NUM_WINDOWS = (BIGINT_WORDS * 32 + WINDOW_SIZE - 1) / WINDOW_SIZE; // ceil(256 / 4) = 64

    ECPointJac res;
    point_set_infinity_jac(&res);  // Initialize result to point at infinity

    for (int window = NUM_WINDOWS - 1; window >= 0; window--) {
        // Perform WINDOW_SIZE doublings per window
        #pragma unroll
        for (int j = 0; j < WINDOW_SIZE; j++) {
            double_point_jac(&res, &res);
        }

        // Extract window bits
        int bit_index = window * WINDOW_SIZE;
        int word_idx = bit_index >> 5;        // bit_index / 32
        int bit_offset = bit_index & 31;      // bit_index % 32

        int window_value = 0;
        if (word_idx < BIGINT_WORDS) {
            if (bit_offset + WINDOW_SIZE <= 32) {
                // All bits in one word
                window_value = (scalar->data[word_idx] >> bit_offset) & ((1U << WINDOW_SIZE) - 1);
            } else {
                // Bits span two words
                int bits_in_first = 32 - bit_offset;
                int bits_in_second = WINDOW_SIZE - bits_in_first;

                uint32_t part1 = scalar->data[word_idx] >> bit_offset;
                uint32_t part2 = 0;
                if (word_idx + 1 < BIGINT_WORDS) {
                    part2 = scalar->data[word_idx + 1] & ((1U << bits_in_second) - 1);
                }
                window_value = (part2 << bits_in_first) | part1;
            }
        }

        // Add from precomputed table if window_value is non-zero
        if (window_value > 0) {
            add_point_jac(&res, &res, &G_precomp[window_value]);
        }
    }

    point_copy_jac(result, &res);
}


__global__ void precompute_G_kernel() {
    if (threadIdx.x == 0) {
        point_set_infinity_jac(&G_precomp[0]);
        point_copy_jac(&G_precomp[1], &const_G_jacobian);
        for (int i = 2; i < (1 << WINDOW_SIZE); i++) {
            add_point_jac(&G_precomp[i], &G_precomp[i-1], &const_G_jacobian);
        }
    }
}