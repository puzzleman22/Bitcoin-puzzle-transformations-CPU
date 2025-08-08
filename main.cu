#include "secp256k1.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <fstream>
#pragma once
#include <stdint.h>
#include <curand_kernel.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Optimized rotate right for SHA-256
__device__ inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ void sha256(const uint8_t* data, int len, uint8_t hash[32]) {
    const uint32_t K[] = {
        0x428a2f98ul,0x71374491ul,0xb5c0fbcful,0xe9b5dba5ul,
        0x3956c25bul,0x59f111f1ul,0x923f82a4ul,0xab1c5ed5ul,
        0xd807aa98ul,0x12835b01ul,0x243185beul,0x550c7dc3ul,
        0x72be5d74ul,0x80deb1feul,0x9bdc06a7ul,0xc19bf174ul,
        0xe49b69c1ul,0xefbe4786ul,0x0fc19dc6ul,0x240ca1ccul,
        0x2de92c6ful,0x4a7484aaul,0x5cb0a9dcul,0x76f988daul,
        0x983e5152ul,0xa831c66dul,0xb00327c8ul,0xbf597fc7ul,
        0xc6e00bf3ul,0xd5a79147ul,0x06ca6351ul,0x14292967ul,
        0x27b70a85ul,0x2e1b2138ul,0x4d2c6dfcul,0x53380d13ul,
        0x650a7354ul,0x766a0abbul,0x81c2c92eul,0x92722c85ul,
        0xa2bfe8a1ul,0xa81a664bul,0xc24b8b70ul,0xc76c51a3ul,
        0xd192e819ul,0xd6990624ul,0xf40e3585ul,0x106aa070ul,
        0x19a4c116ul,0x1e376c08ul,0x2748774cul,0x34b0bcb5ul,
        0x391c0cb3ul,0x4ed8aa4aul,0x5b9cca4ful,0x682e6ff3ul,
        0x748f82eeul,0x78a5636ful,0x84c87814ul,0x8cc70208ul,
        0x90befffaul,0xa4506cebul,0xbef9a3f7ul,0xc67178f2ul
    };

    uint32_t h[8] = {
        0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul,
        0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul
    };

    // Optimized for 33-byte input (compressed pubkey)
    uint8_t full[64] = {0};
    
    // Copy input data
    #pragma unroll
    for (int i = 0; i < len; ++i) full[i] = data[i];
    full[len] = 0x80;
    
    // Add length in bits (big-endian) at the end
    uint64_t bit_len = (uint64_t)len * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        full[63 - i] = bit_len >> (8 * i);
    }

    // Process single block (we know it's only one block for 33 bytes)
    uint32_t w[64];
    
    // Load message schedule with proper byte order
    #pragma unroll 16
    for (int i = 0; i < 16; ++i) {
        w[i] = (full[4 * i] << 24) | (full[4 * i + 1] << 16) |
               (full[4 * i + 2] << 8) | full[4 * i + 3];
    }
    
    // Extend message schedule
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hval = h[7];

    // Main compression loop
    #pragma unroll 8
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = hval + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        hval = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hval;

    // Output hash (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        hash[4 * i + 0] = (h[i] >> 24) & 0xFF;
        hash[4 * i + 1] = (h[i] >> 16) & 0xFF;
        hash[4 * i + 2] = (h[i] >> 8) & 0xFF;
        hash[4 * i + 3] = (h[i] >> 0) & 0xFF;
    }
}

__device__ void ripemd160(const uint8_t* msg, uint8_t* out) {
    // RIPEMD-160 constants
    const uint32_t K1[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
    const uint32_t K2[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};
    
    // Message schedule for left and right lines
    const int ZL[80] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
        3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
        1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
        4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
    };
    
    const int ZR[80] = {
        5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
        6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
        15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
        8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
        12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
    };
    
    // Shift amounts for left and right lines
    const int SL[80] = {
        11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
        7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
        11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
        11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
        9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
    };
    
    const int SR[80] = {
        8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
        9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
        9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
        15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
        8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
    };
    
    // Initialize hash values
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;
    
    // Prepare message: add padding and length
    uint8_t buffer[64];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        buffer[i] = msg[i];
    }
    
    // Add padding
    buffer[32] = 0x80;
    #pragma unroll
    for (int i = 33; i < 56; i++) {
        buffer[i] = 0x00;
    }
    
    // Add length (256 bits = 32 bytes)
    uint64_t bitlen = 256;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        buffer[56 + i] = (bitlen >> (i * 8)) & 0xFF;
    }
    
    // Convert buffer to 32-bit data (little-endian)
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        X[i] = ((uint32_t)buffer[i*4]) | 
               ((uint32_t)buffer[i*4 + 1] << 8) | 
               ((uint32_t)buffer[i*4 + 2] << 16) | 
               ((uint32_t)buffer[i*4 + 3] << 24);
    }
    
    // Working variables
    uint32_t AL = h0, BL = h1, CL = h2, DL = h3, EL = h4;
    uint32_t AR = h0, BR = h1, CR = h2, DR = h3, ER = h4;
    
    // Process message in 5 rounds of 16 operations each
    #pragma unroll 10
    for (int j = 0; j < 80; j++) {
        uint32_t T;
        
        // Left line
        if (j < 16) {
            T = AL + (BL ^ CL ^ DL) + X[ZL[j]] + K1[0];
        } else if (j < 32) {
            T = AL + ((BL & CL) | (~BL & DL)) + X[ZL[j]] + K1[1];
        } else if (j < 48) {
            T = AL + ((BL | ~CL) ^ DL) + X[ZL[j]] + K1[2];
        } else if (j < 64) {
            T = AL + ((BL & DL) | (CL & ~DL)) + X[ZL[j]] + K1[3];
        } else {
            T = AL + (BL ^ (CL | ~DL)) + X[ZL[j]] + K1[4];
        }
        T = ((T << SL[j]) | (T >> (32 - SL[j]))) + EL;
        AL = EL; EL = DL; DL = (CL << 10) | (CL >> 22); CL = BL; BL = T;
        
        // Right line
        if (j < 16) {
            T = AR + (BR ^ (CR | ~DR)) + X[ZR[j]] + K2[0];
        } else if (j < 32) {
            T = AR + ((BR & DR) | (CR & ~DR)) + X[ZR[j]] + K2[1];
        } else if (j < 48) {
            T = AR + ((BR | ~CR) ^ DR) + X[ZR[j]] + K2[2];
        } else if (j < 64) {
            T = AR + ((BR & CR) | (~BR & DR)) + X[ZR[j]] + K2[3];
        } else {
            T = AR + (BR ^ CR ^ DR) + X[ZR[j]] + K2[4];
        }
        T = ((T << SR[j]) | (T >> (32 - SR[j]))) + ER;
        AR = ER; ER = DR; DR = (CR << 10) | (CR >> 22); CR = BR; BR = T;
    }
    
    // Add results
    uint32_t T = h1 + CL + DR;
    h1 = h2 + DL + ER;
    h2 = h3 + EL + AR;
    h3 = h4 + AL + BR;
    h4 = h0 + BL + CR;
    h0 = T;
    
    // Convert hash to bytes (little-endian)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i]      = (h0 >> (i * 8)) & 0xFF;
        out[i + 4]  = (h1 >> (i * 8)) & 0xFF;
        out[i + 8]  = (h2 >> (i * 8)) & 0xFF;
        out[i + 12] = (h3 >> (i * 8)) & 0xFF;
        out[i + 16] = (h4 >> (i * 8)) & 0xFF;
    }
}

__device__ __forceinline__ void hash160(const uint8_t* data, int len, uint8_t out[20]) {
    uint8_t sha[32];
    sha256(data, len, sha);
    ripemd160(sha, out);
}

void init_gpu_constants() {
    // 1) 定义 p_host
    const BigInt p_host = {
        { 0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
          0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF }
    };
    // 2) 定义 G_jacobian_host
    const ECPointJac G_jacobian_host = {
        {{ 0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
                0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E }},
        {{ 0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
                0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77 }},
        {{ 1, 0, 0, 0, 0, 0, 0, 0 }}
    };
    // 3) 定义 n_host
    const BigInt n_host = {
        { 0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
          0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF }
    };

    // 然后再复制到 __constant__ 内存
    CHECK_CUDA(cudaMemcpyToSymbol(const_p, &p_host, sizeof(BigInt)));
    CHECK_CUDA(cudaMemcpyToSymbol(const_G_jacobian, &G_jacobian_host, sizeof(ECPointJac)));
    CHECK_CUDA(cudaMemcpyToSymbol(const_n, &n_host, sizeof(BigInt)));
}

std::string generate_random_hex(size_t length) {
    const char hex_chars[] = "0123456789abcdef";
    std::string hex_string;
    hex_string.reserve(length); // Reserve space to avoid reallocations

    for (size_t i = 0; i < length; ++i) {
        hex_string += hex_chars[rand() % 16];
    }

    return hex_string;
}

std::string zfill(const std::string& input, size_t total_length) {
    if (input.length() >= total_length) {
        return input;
    }
    return std::string(total_length - input.length(), '0') + input;
}

__device__ __forceinline__ uint8_t get_byte(const BigInt& a, int i) {
    // Convert to big-endian byte order
    int word_index = 7 - (i / 4);       // reverse word order
    int byte_index = 3 - (i % 4);       // reverse byte order within word
    return (a.data[word_index] >> (8 * byte_index)) & 0xFF;
}

__device__ __forceinline__ void coords_to_compressed_pubkey(const BigInt& x, const BigInt& y, uint8_t* pubkey) {
    // Prefix: 0x02 if y is even, 0x03 if y is odd
    pubkey[0] = (y.data[0] & 1) ? 0x03 : 0x02;

    // Copy x coordinate (32 bytes) with unrolling
    #pragma unroll 8
    for (int i = 0; i < 32; i++) {
        pubkey[1 + i] = get_byte(x, i);
    }
}

// Convert a hex character to its numeric value
__device__ __forceinline__ uint8_t hex_char_to_byte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0; // Invalid character
}

// Convert hex string to bytes
__device__ void hex_string_to_bytes(const char* hex_str, uint8_t* bytes, int num_bytes) {
    #pragma unroll 8
    for (int i = 0; i < num_bytes; i++) {
        bytes[i] = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                   hex_char_to_byte(hex_str[i * 2 + 1]);
    }
}

// Compare two hash160 arrays
__device__ __forceinline__ bool compare_hash160(const uint8_t* hash1, const uint8_t* hash2) {
    // Use 32-bit comparisons for speed
    uint32_t* h1 = (uint32_t*)hash1;
    uint32_t* h2 = (uint32_t*)hash2;
    
    return (h1[0] == h2[0]) && (h1[1] == h2[1]) && 
           (h1[2] == h2[2]) && (h1[3] == h2[3]) && 
           (h1[4] == h2[4]);
}

// Compare hash160 with hex string directly - optimized version
__device__ __forceinline__ bool compare_hash160_with_hex(const uint8_t* hash, const char* hex_str) {
    // Early exit on first mismatch
    #pragma unroll 5
    for (int i = 0; i < 20; i++) {
        uint8_t byte = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                       hex_char_to_byte(hex_str[i * 2 + 1]);
        if (hash[i] != byte) {
            return false;
        }
    }
    return true;
}

// Compare two BigInts: returns -1 if a < b, 0 if a == b, 1 if a > b
__device__ __forceinline__ int bigint_compare(const BigInt* a, const BigInt* b) {
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

// Subtract b from a (a must be >= b), result stored in result
__device__ void bigint_subtract(const BigInt* a, const BigInt* b, BigInt* result) {
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t temp = (uint64_t)a->data[i] - b->data[i] - borrow;
        result->data[i] = (uint32_t)temp;
        borrow = (temp >> 32) & 1;
    }
}

// Add b to a, result stored in result
__device__ void bigint_add(const BigInt* a, const BigInt* b, BigInt* result) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t temp = (uint64_t)a->data[i] + b->data[i] + carry;
        result->data[i] = (uint32_t)temp;
        carry = temp >> 32;
    }
}

// Fast PRNG with better mixing
__device__ __forceinline__ uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

__device__ inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}


// Method 0: Chaotic mixing
__device__ void generate_method_chaotic(uint64_t* rng_state, const BigInt* min, const BigInt* range_plus_one, 
                                       int highest_word, int highest_bit, BigInt* result) {
    uint32_t chaos_counter = 0;
    
    while (true) {
        BigInt candidate = {0};
        uint64_t mixer = xorshift64(rng_state);
        
        #pragma unroll 4
        for (int i = 0; i <= highest_word && i < 4; i++) {
            uint64_t r1 = xorshift64(rng_state);
            uint64_t r2 = xorshift64(rng_state);
            
            r1 ^= (r2 << 17) | (r2 >> 47);
            r2 = r1 * 0x9E3779B97F4A7C15ULL;
            r1 = (r1 >> 30) ^ r2;
            
            r1 ^= chaos_counter * 0xDEADBEEF;
            r2 = rotl64(r2, chaos_counter & 63);
            
            if (i * 2 < 8) candidate.data[i * 2] = (uint32_t)(r1 ^ mixer);
            if (i * 2 + 1 < 8) candidate.data[i * 2 + 1] = (uint32_t)((r2 >> 32) ^ (mixer >> 32));
            
            mixer = rotl64(mixer, 13);
        }
        
        for (int i = 0; i <= highest_word && i < 8; i++) {
            uint32_t val = candidate.data[i];
            val ^= (val >> 7) ^ (val << 25);
            val *= 0x85EBCA6B;
            val ^= (val >> 13) ^ (val << 19);
            candidate.data[i] = val;
        }
        
        if (highest_word < 8) {
            uint32_t mask = (highest_bit == 32) ? 0xFFFFFFFF : ((1U << highest_bit) - 1);
            candidate.data[highest_word] &= mask;
            #pragma unroll
            for (int i = highest_word + 1; i < 8; i++) {
                candidate.data[i] = 0;
            }
        }
        
        chaos_counter++;
        
        if (bigint_compare(&candidate, range_plus_one) < 0) {
            bigint_add(&candidate, min, result);
            return;
        }
    }
}

// Method 1: Thread-based entropy
__device__ void generate_method_entropy(uint64_t* rng_state, const BigInt* min, const BigInt* range_plus_one, 
                                       int highest_word, int highest_bit, BigInt* result, int tid) {
    uint32_t thread_entropy = tid ^ (blockIdx.x << 10) ^ (blockIdx.y << 20);
    uint64_t time_based = clock64();
    
    while (true) {
        BigInt candidate = {0};
        
        uint64_t pcg_state = *rng_state;
        *rng_state = pcg_state * 0x5851F42D4C957F2DULL + thread_entropy;
        uint64_t xor_shifted = ((pcg_state >> 18) ^ pcg_state) >> 27;
        uint32_t rot = pcg_state >> 59;
        uint64_t pcg_output = (xor_shifted >> rot) | (xor_shifted << ((~rot + 1) & 31));
        
        for (int i = 0; i <= highest_word; i++) {
            uint64_t r1 = xorshift64(rng_state);
            uint64_t r2 = pcg_output;
            uint64_t r3 = time_based;
            
            r1 ^= rotl64(r2, 23) ^ rotl64(r3, 41);
            r2 = (r1 * 0x9E3779B97F4A7C15ULL) ^ thread_entropy;
            r3 = (r2 + r3) * 0xC45979A72B4C8A7FULL;
            
            if (i < 8) {
                candidate.data[i] = (uint32_t)(r1 ^ r2 ^ r3);
            }
            
            pcg_output = r3;
            time_based = rotl64(time_based, 7) ^ r1;
        }
        
        if (highest_word < 8) {
            uint32_t mask = (highest_bit == 32) ? 0xFFFFFFFF : ((1U << highest_bit) - 1);
            candidate.data[highest_word] &= mask;
            #pragma unroll
            for (int i = highest_word + 1; i < 8; i++) {
                candidate.data[i] = 0;
            }
        }
        
        if (bigint_compare(&candidate, range_plus_one) < 0) {
            bigint_add(&candidate, min, result);
            return;
        }
    }
}

// Method 2: Avalanche effect
__device__ void generate_method_avalanche(uint64_t* rng_state, const BigInt* min, const BigInt* range_plus_one, 
                                         int highest_word, int highest_bit, BigInt* result) {
    uint64_t avalanche[4];
    avalanche[0] = xorshift64(rng_state);
    avalanche[1] = xorshift64(rng_state);
    avalanche[2] = xorshift64(rng_state);
    avalanche[3] = xorshift64(rng_state);
    
    while (true) {
        BigInt candidate = {0};
        
        for (int round = 0; round < 3; round++) {
            avalanche[0] ^= avalanche[3] << 13;
            avalanche[1] ^= avalanche[0] >> 7;
            avalanche[2] ^= avalanche[1] << 17;
            avalanche[3] ^= avalanche[2] >> 11;
            
            uint64_t temp = avalanche[0];
            avalanche[0] = avalanche[1] ^ (avalanche[2] * 0x9E3779B97F4A7C15ULL);
            avalanche[1] = avalanche[2] + rotl64(avalanche[3], 23);
            avalanche[2] = avalanche[3] ^ temp;
            avalanche[3] = temp + avalanche[0];
        }
        
        for (int i = 0; i <= highest_word && i < 8; i++) {
            uint64_t val = avalanche[i & 3];
            
            val ^= val >> 33;
            val *= 0xFF51AFD7ED558CCDULL;
            val ^= val >> 33;
            val *= 0xC4CEB9FE1A85EC53ULL;
            val ^= val >> 33;
            
            candidate.data[i] = (uint32_t)val;
            avalanche[i & 3] = xorshift64(rng_state) ^ val;
        }
        
        if (highest_word < 8) {
            uint32_t mask = (highest_bit == 32) ? 0xFFFFFFFF : ((1U << highest_bit) - 1);
            candidate.data[highest_word] &= mask;
            #pragma unroll
            for (int i = highest_word + 1; i < 8; i++) {
                candidate.data[i] = 0;
            }
        }
        
        if (bigint_compare(&candidate, range_plus_one) < 0) {
            bigint_add(&candidate, min, result);
            return;
        }
    }
}


__device__ void generate_random_bigint_range_fast(uint64_t* rng_state, const BigInt* min, const BigInt* max, BigInt* result) {
    // Calculate range = max - min + 1
    BigInt range, range_plus_one;
    bigint_subtract(max, min, &range);
    
    BigInt one = {1, 0, 0, 0, 0, 0, 0, 0};
    bigint_add(&range, &one, &range_plus_one);

    // Find highest bit
    int highest_word = 7;
    while (highest_word > 0 && range_plus_one.data[highest_word] == 0)
        highest_word--;

    int highest_bit = 31;
    while (highest_bit > 0 && (range_plus_one.data[highest_word] & (1U << highest_bit)) == 0)
        highest_bit--;
    highest_bit += 1;

    // Generate random using xorshift
    while (true) {
        BigInt candidate = {0};

        // Fill with random data - optimized loop
        #pragma unroll 4
        for (int i = 0; i <= highest_word && i < 4; i++) {
            uint64_t r = xorshift64(rng_state);
            if (i * 2 < 8) candidate.data[i * 2] = (uint32_t)r;
            if (i * 2 + 1 < 8) candidate.data[i * 2 + 1] = (uint32_t)(r >> 32);
        }

        // Mask highest word
        if (highest_word < 8) {
            uint32_t mask = (highest_bit == 32) ? 0xFFFFFFFF : ((1U << highest_bit) - 1);
            candidate.data[highest_word] &= mask;
            
            #pragma unroll
            for (int i = highest_word + 1; i < 8; i++) {
                candidate.data[i] = 0;
            }
        }

        // Check if in range
        if (bigint_compare(&candidate, &range_plus_one) < 0) {
            bigint_add(&candidate, min, result);
            return;
        }
    }
}

// Convert hex string to BigInt - optimized
__device__ void hex_to_bigint(const char* hex_str, BigInt* bigint) {
    // Initialize all data to 0
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }
    
    int len = 0;
    while (hex_str[len] != '\0' && len < 64) len++;
    
    // Process hex string from right to left
    int word_idx = 0;
    int bit_offset = 0;
    
    for (int i = len - 1; i >= 0 && word_idx < 8; i--) {
        uint8_t val = hex_char_to_byte(hex_str[i]);
        
        bigint->data[word_idx] |= ((uint32_t)val << bit_offset);
        
        bit_offset += 4;
        if (bit_offset >= 32) {
            bit_offset = 0;
            word_idx++;
        }
    }
}

// Convert BigInt to hex string - optimized
__device__ void bigint_to_hex(const BigInt* bigint, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int idx = 0;
    bool leading_zero = true;
    
    // Process from most significant word to least
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        for (int j = 28; j >= 0; j -= 4) {
            uint8_t nibble = (bigint->data[i] >> j) & 0xF;
            if (nibble != 0 || !leading_zero || (i == 0 && j == 0)) {
                hex_str[idx++] = hex_chars[nibble];
                leading_zero = false;
            }
        }
    }
    
    // Handle case where number is 0
    if (idx == 0) {
        hex_str[idx++] = '0';
    }
    
    hex_str[idx] = '\0';
}

// Optimized byte to hex conversion
__device__ __forceinline__ void byte_to_hex(uint8_t byte, char* out) {
    const char hex_chars[] = "0123456789abcdef";
    out[0] = hex_chars[(byte >> 4) & 0xF];
    out[1] = hex_chars[byte & 0xF];
}

__device__ void hash160_to_hex(uint8_t* hash, char* hex_str) {
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        byte_to_hex(hash[i], &hex_str[i * 2]);
    }
    hex_str[40] = '\0';
}

#define HEX_LENGTH 64  // 64 hex characters

// Optimized hex rotation functions
__device__ __forceinline__ void hex_rotate_right_by_one(char* hex_str) {
    int actual_length = 0;
    #pragma unroll 8
    for (int i = 0; i < HEX_LENGTH; i++) {
        if (hex_str[i] == '\0') {
            actual_length = i;
            break;
        }
    }
    if (actual_length == 0) {
        actual_length = HEX_LENGTH;
    }
    
    if (actual_length <= 1) return;
    
    // Find the first occurrence of '1'
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (hex_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    int rotation_start = first_one + 1;
    int rotation_length = actual_length - rotation_start;
    
    if (rotation_length <= 1) return;
    
    char last_char = hex_str[rotation_start + rotation_length - 1];
    
    // Use memmove for better performance
    for (int i = rotation_length - 1; i > 0; i--) {
        hex_str[rotation_start + i] = hex_str[rotation_start + i - 1];
    }
    
    hex_str[rotation_start] = last_char;
}

__device__ __forceinline__ void hex_rotate_left_by_one(char* hex_str) {
    int actual_length = 0;
    #pragma unroll 8
    for (int i = 0; i < HEX_LENGTH; i++) {
        if (hex_str[i] == '\0') {
            actual_length = i;
            break;
        }
    }
    if (actual_length == 0) {
        actual_length = HEX_LENGTH;
    }
    
    if (actual_length <= 1) return;
    
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (hex_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    int rotation_start = first_one + 1;
    int rotation_length = actual_length - rotation_start;
    
    if (rotation_length <= 1) return;
    
    char first_char = hex_str[rotation_start];
    
    for (int i = 0; i < rotation_length - 1; i++) {
        hex_str[rotation_start + i] = hex_str[rotation_start + i + 1];
    }
    
    hex_str[rotation_start + rotation_length - 1] = first_char;
}

// Use lookup table for hex increment/decrement
__constant__ char hex_inc_table[16] = {'1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','0'};
__constant__ char hex_dec_table[16] = {'f','0','1','2','3','4','5','6','7','8','9','a','b','c','d','e'};

__device__ __forceinline__ char hex_increment(char c) {
    if (c >= '0' && c <= '9') return hex_inc_table[c - '0'];
    if (c >= 'a' && c <= 'f') return hex_inc_table[c - 'a' + 10];
    if (c >= 'A' && c <= 'F') return hex_inc_table[c - 'A' + 10];
    return c;
}

__device__ __forceinline__ char hex_decrement(char c) {
    if (c >= '0' && c <= '9') return hex_dec_table[c - '0'];
    if (c >= 'a' && c <= 'f') return hex_dec_table[c - 'a' + 10];
    if (c >= 'A' && c <= 'F') return hex_dec_table[c - 'A' + 10];
    return c;
}

__device__ __forceinline__ void hex_vertical_rotate_up(char* hex_str) {
    int actual_length = 0;
    #pragma unroll 8
    for (int i = 0; i < HEX_LENGTH; i++) {
        if (hex_str[i] == '\0') {
            actual_length = i;
            break;
        }
    }
    if (actual_length == 0) actual_length = HEX_LENGTH;
    
    if (actual_length <= 1) return;
    
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (hex_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    // Rotate all characters after the first '1' vertically up
    #pragma unroll 8
    for (int i = first_one + 1; i < actual_length; i++) {
        hex_str[i] = hex_increment(hex_str[i]);
    }
}

__device__ __forceinline__ void hex_vertical_rotate_down(char* hex_str) {
    int actual_length = 0;
    #pragma unroll 8
    for (int i = 0; i < HEX_LENGTH; i++) {
        if (hex_str[i] == '\0') {
            actual_length = i;
            break;
        }
    }
    if (actual_length == 0) actual_length = HEX_LENGTH;
    
    if (actual_length <= 1) return;
    
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (hex_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    #pragma unroll 8
    for (int i = first_one + 1; i < actual_length; i++) {
        hex_str[i] = hex_decrement(hex_str[i]);
    }
}

__device__ void leftPad64(char* output, const char* suffix) {
    int suffix_len = 0;
    // Get length of suffix
    while (suffix[suffix_len] != '\0' && suffix_len < 64) {
        ++suffix_len;
    }

    int pad_len = 64 - suffix_len;

    // Fill left padding with '0' using memset
    #pragma unroll 8
    for (int i = 0; i < pad_len; ++i) {
        output[i] = '0';
    }

    // Copy suffix to the right
    #pragma unroll 8
    for (int i = 0; i < suffix_len; ++i) {
        output[pad_len + i] = suffix[i];
    }

    output[64] = '\0';
}

__device__ __forceinline__ int str_len(const char* str) {
    int len = 0;
    while (str[len] != '\0') {
        ++len;
    }
    return len;
}

__device__ void reverseAfterFirst1(char* hex) {
    // Find first '1'
    char* first1 = hex;
    while (*first1 && *first1 != '1') first1++;
    
    if (*first1 == '\0' || *(first1 + 1) == '\0') return;
    
    // Find end
    char* end = first1 + 1;
    while (*end) end++;
    end--;
    
    // Reverse after '1'
    char* start = first1 + 1;
    while (start < end) {
        char temp = *start;
        *start = *end;
        *end = temp;
        start++;
        end--;
    }
}

__device__ void invertHexAfterFirst1(char* hex) {
    bool foundFirst1 = false;
    
    for (int i = 0; hex[i] != '\0'; i++) {
        if (!foundFirst1 && hex[i] == '1') {
            foundFirst1 = true;
            continue;
        }
        
        if (foundFirst1) {
            char c = hex[i];
            int val = hex_char_to_byte(c);
            
            // Invert all 4 bits of this hex digit
            val = (~val) & 0xF;
            
            // Convert back to hex char
            hex[i] = (val < 10) ? ('0' + val) : ('a' + (val - 10));
        }
    }
}

__device__ __forceinline__ int d_strlen(const char* str) {
    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

__device__ void incrementBigInt(BigInt* num) {
    // Start from the least significant word (data[0])
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        num->data[i]++;
        
        // If no overflow, we're done
        if (num->data[i] != 0) {
            break;
        }
        // If overflow (wrapped to 0), continue to next word
    }
}

__device__ void clearLowest8Bits(BigInt* num) {
    // Clear the lowest 8 bits of the least significant word
    num->data[0] &= 0xFFFFFF00;
}

__device__ __forceinline__ uint64_t mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// Convert BigInt to binary string
__device__ void bigint_to_binary(const BigInt* bigint, char* binary_str) {
    int idx = 0;
    bool leading_zero = true;
    
    // Process from most significant word to least
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        for (int j = 31; j >= 0; j--) {
            uint8_t bit = (bigint->data[i] >> j) & 1;
            if (bit != 0 || !leading_zero || (i == 0 && j == 0)) {
                binary_str[idx++] = bit ? '1' : '0';
                leading_zero = false;
            }
        }
    }
    
    // Handle case where number is 0
    if (idx == 0) {
        binary_str[idx++] = '0';
    }
    
    binary_str[idx] = '\0';
}

// Convert binary string to BigInt
__device__ void binary_to_bigint(const char* binary_str, BigInt* bigint) {
    // Initialize all data to 0
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }
    
    int len = 0;
    while (binary_str[len] != '\0' && len < 256) len++; // Max 256 bits
    
    // Process binary string from right to left
    int word_idx = 0;
    int bit_offset = 0;
    
    for (int i = len - 1; i >= 0 && word_idx < 8; i--) {
        if (binary_str[i] == '1') {
            bigint->data[word_idx] |= (1U << bit_offset);
        }
        
        bit_offset++;
        if (bit_offset >= 32) {
            bit_offset = 0;
            word_idx++;
        }
    }
}

// Binary rotate left by one (after first '1')
__device__ __forceinline__ void binary_rotate_left_by_one(char* binary_str) {
    int actual_length = 0;
    while (binary_str[actual_length] != '\0' && actual_length < 256) {
        actual_length++;
    }
    
    if (actual_length <= 1) return;
    
    // Find first '1'
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (binary_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    int rotation_start = first_one + 1;
    int rotation_length = actual_length - rotation_start;
    
    if (rotation_length <= 1) return;
    
    char first_char = binary_str[rotation_start];
    
    for (int i = 0; i < rotation_length - 1; i++) {
        binary_str[rotation_start + i] = binary_str[rotation_start + i + 1];
    }
    
    binary_str[rotation_start + rotation_length - 1] = first_char;
}

// Binary vertical rotate up (increment each 4-bit nibble after first '1')
__device__ __forceinline__ void binary_vertical_rotate_up(char* binary_str) {
    int actual_length = 0;
    while (binary_str[actual_length] != '\0' && actual_length < 256) {
        actual_length++;
    }
    
    if (actual_length <= 1) return;
    
    // Find first '1'
    int first_one = -1;
    for (int i = 0; i < actual_length; i++) {
        if (binary_str[i] == '1') {
            first_one = i;
            break;
        }
    }
    
    if (first_one == -1 || first_one >= actual_length - 1) return;
    
    // Pad length to multiple of 4 for nibble processing
    int start_pos = first_one + 1;
    
    // Process each 4-bit nibble after the first '1'
    for (int nibble_start = start_pos; nibble_start < actual_length; nibble_start += 4) {
        // Extract current nibble (up to 4 bits)
        uint8_t nibble_val = 0;
        int nibble_size = 0;
        
        // Read nibble value
        for (int bit = 0; bit < 4 && (nibble_start + bit) < actual_length; bit++) {
            if (binary_str[nibble_start + bit] == '1') {
                nibble_val |= (1 << (3 - bit));
            }
            nibble_size++;
        }
        
        // Increment nibble (with wrap-around: F -> 0)
        nibble_val = (nibble_val + 1) & 0xF;
        
        // Write back the incremented nibble
        for (int bit = 0; bit < nibble_size; bit++) {
            binary_str[nibble_start + bit] = ((nibble_val >> (3 - bit)) & 1) ? '1' : '0';
        }
    }
}


// Reverse binary string after first '1'
__device__ void reverseBinaryAfterFirst1(char* binary_str) {
    // Find first '1'
    char* first1 = binary_str;
    while (*first1 && *first1 != '1') first1++;
    
    if (*first1 == '\0' || *(first1 + 1) == '\0') return;
    
    // Find end
    char* end = first1 + 1;
    while (*end) end++;
    end--;
    
    // Reverse after '1'
    char* start = first1 + 1;
    while (start < end) {
        char temp = *start;
        *start = *end;
        *end = temp;
        start++;
        end--;
    }
}

// Invert binary string after first '1'
__device__ void invertBinaryAfterFirst1(char* binary_str) {
    bool foundFirst1 = false;
    
    for (int i = 0; binary_str[i] != '\0'; i++) {
        if (!foundFirst1 && binary_str[i] == '1') {
            foundFirst1 = true;
            continue;
        }
        
        if (foundFirst1) {
            // Invert bit
            binary_str[i] = (binary_str[i] == '0') ? '1' : '0';
        }
    }
}

// Convert binary string to hex string
__device__ void binary_to_hex(const char* binary_str, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int binary_len = 0;
    
    // Get binary string length
    while (binary_str[binary_len] != '\0') binary_len++;
    
    // Pad binary string to multiple of 4 bits
    int padded_len = ((binary_len + 3) / 4) * 4;
    
    int hex_idx = 0;
    bool leading_zero = true;
    
    // Process 4 bits at a time
    for (int i = 0; i < padded_len; i += 4) {
        uint8_t nibble = 0;
        
        // Convert 4 binary digits to hex nibble
        for (int j = 0; j < 4; j++) {
            int bit_pos = i + j;
            int actual_pos = bit_pos - (padded_len - binary_len);
            
            if (actual_pos >= 0 && actual_pos < binary_len && binary_str[actual_pos] == '1') {
                nibble |= (1 << (3 - j));
            }
        }
        
        // Skip leading zeros
        if (nibble != 0 || !leading_zero || i >= padded_len - 4) {
            hex_str[hex_idx++] = hex_chars[nibble];
            leading_zero = false;
        }
    }
    
    // Handle case where result is 0
    if (hex_idx == 0) {
        hex_str[hex_idx++] = '0';
    }
    
    hex_str[hex_idx] = '\0';
}

// Optimization 4: Direct binary to BigInt conversion (implement this helper)
__device__ void binary_to_bigint_direct(const char* binary, BigInt* result) {
    // Initialize result to zero
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result->data[i] = 0;
    }
    
    // Process binary string directly without hex intermediate
    int len = d_strlen(binary);
    for (int i = 0; i < len && i < 256; i++) {
        if (binary[len - 1 - i] == '1') {
            int word_idx = i >> 5; // i / 32
            int bit_idx = i & 31;  // i % 32
            if (word_idx < BIGINT_WORDS) {
                result->data[word_idx] |= (1U << bit_idx);
            }
        }
    }
}

// Optimization 5: Faster hash160 comparison (implement this)
__device__ __forceinline__ bool compare_hash160_fast(const uint8_t* hash1, const uint8_t* hash2) {
    // Use 64-bit comparisons instead of byte-by-byte
    const uint64_t* h1 = (const uint64_t*)hash1;
    const uint64_t* h2 = (const uint64_t*)hash2;
    
    return (h1[0] == h2[0]) && (h1[1] == h2[1]) && 
           (*(uint32_t*)(hash1 + 16) == *(uint32_t*)(hash2 + 16));
}


__device__ int BIGINT_SIZE = 8;
__device__ volatile int g_found = 0;
__device__ char g_found_hex[65] = {0};        // Original hex
__device__ char g_found_hash160[41] = {0};    // Hash160 result

// Simplified function to preserve "1" prefix and rotate only the part after "1"
__device__ void create_base_variation_preserve_prefix(const BigInt* base_num, BigInt* result, int thread_base, int rotation) {
    // Convert to hex string
    char hex_str[65];
    bigint_to_hex(base_num, hex_str);
    
    // Find the position of the first "1" in the hex string
    int first_one_pos = -1;
    int hex_len = 0;
    
    // Get length and find first "1"
    while (hex_str[hex_len] != '\0') {
        if (hex_str[hex_len] == '1' && first_one_pos == -1) {
            first_one_pos = hex_len;
        }
        hex_len++;
    }
    
    // If no "1" found, just copy original and apply minimal change
    if (first_one_pos == -1) {
        copy_bigint(result, base_num);
        return;
    }
    
    // Create rotated version - preserve everything up to and including first "1"
    char rotated_hex[65];
    
    // Copy the prefix (up to and including the first "1")
    int i;
    for (i = 0; i <= first_one_pos; i++) {
        rotated_hex[i] = hex_str[i];
    }
    
    // Get the part after "1" that needs to be rotated
    char *after_one = &hex_str[first_one_pos + 1];
    int after_one_len = hex_len - first_one_pos - 1;
    
    if (after_one_len > 0) {
        // Apply rotation to the part after "1"
        int shift = (thread_base + rotation) % after_one_len;
        if (shift == 0) shift = 1; // Ensure we always do some rotation
        
        // Rotate the substring after "1"
        for (int j = 0; j < after_one_len; j++) {
            rotated_hex[first_one_pos + 1 + j] = after_one[(j + shift) % after_one_len];
        }
    }
    
    // Null terminate
    rotated_hex[hex_len] = '\0';
    
    // Convert back to BigInt
    hex_to_bigint(rotated_hex, result);
}

// Alternative simpler base conversion that works with existing functions
__device__ void simple_base_transform(BigInt* num, int base, int rotation) {
    // Apply transformations based on base and rotation without string conversion
    
    // Rotation-based bit shifting
    for (int r = 0; r < rotation; r++) {
        // Rotate bits based on base
        uint32_t bits_to_rotate = base % 32;
        
        for (int shift = 0; shift < bits_to_rotate; shift++) {
            uint32_t carry = 0;
            // Right rotation
            for (int i = BIGINT_SIZE - 1; i >= 0; i--) {
                uint32_t new_carry = num->data[i] & 1;
                num->data[i] = (num->data[i] >> 1) | (carry << 31);
                carry = new_carry;
            }
            num->data[BIGINT_SIZE - 1] |= (carry << 31);
        }
    }
    
    // Base-specific transformation
    if (base % 2 == 0) {
        // Even bases: XOR with pattern
        num->data[0] ^= (base << 16) | rotation;
    } else {
        // Odd bases: Add pattern
        BigInt temp_add;
        for (int i = 0; i < BIGINT_SIZE; i++) {
            temp_add.data[i] = 0;
        }
        temp_add.data[0] = (base * rotation) & 0xFFFFFFFF;
        bigint_add(num, num, &temp_add);
    }
}

// Optimization: Assign bases per warp (32 threads)
__global__ void start_optimized(const char* minRangePure, const char* maxRangePure, const char* target) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("min: %s\n", minRangePure);
        printf("max: %s\n", maxRangePure);
        printf("ripemd160 target: %s\n\n", target);
    }
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int original_length = d_strlen(minRangePure) * 4;
    
    // Calculate warp ID and thread's position within the warp
    int warp_id = tid / 32;  // Which warp this thread belongs to
    int lane_id = threadIdx.x % 32;  // Thread's position within its warp (0-31)
    
    // Each warp gets a different base (2-32)
    // Cycle through bases if we have more than 31 warps
    int warp_base = 2 + (warp_id % 31);  // Bases 2-32
    
    // Move these outside the loop - they don't change
    char minRange[65];
    leftPad64(minRange, minRangePure);
    char maxRange[65];
    leftPad64(maxRange, maxRangePure);
    
    // Convert min/max to BigInt once outside the loop
    BigInt min, max;
    hex_to_bigint(minRange, &min);
    hex_to_bigint(maxRange, &max);
    
    // Performance tracking variables
    unsigned long long local_keys_checked = 0;
    
    // Improved seeding with better entropy mixing
    // Use both warp_id and lane_id for better entropy distribution
    uint64_t seed = clock64();
    seed ^= ((uint64_t)warp_id << 40) | ((uint64_t)lane_id << 32) | 
            ((uint64_t)blockIdx.x << 16) | threadIdx.x;
    seed = mix(seed);
    seed ^= ((uint64_t)gridDim.x << 48) | ((uint64_t)blockDim.x << 32);
    uint64_t rng_state = mix(seed);
    
    // Pre-allocate ALL working variables once - avoid repeated allocation overhead
    BigInt random_value, priv2, priv, base_variant;
    ECPointJac result_jac;
    ECPoint public_key;
    uint8_t pubkey[33];
    uint8_t hash160_out[20];
    char hash160_str[41];
    char temp_hex[65];
    char binary[257];
    int local_found = 0;
    
    // Pre-convert target once
    uint8_t target_bytes[20];
    hex_string_to_bytes(target, target_bytes, 20);
    
    // Debug output for first thread of each warp
    if (lane_id == 0 && warp_id < 5) {
        printf("Warp %d using base %d\n", warp_id, warp_base);
    }
    
    int c = 0;
    while(local_found == 0 && g_found == 0) {
        // 1) Generate the base number
        // Each thread in the warp generates its own random value
        generate_random_bigint_range_fast(&rng_state, &min, &max, &random_value);
        
        // Debug output for first thread of first warp
        if (warp_id == 0 && lane_id == 0 && c < 3) {
            bigint_to_hex(&random_value, temp_hex);
            printf("Warp %d, Lane %d (base %d): Original hex: %s\n", 
                   warp_id, lane_id, warp_base, temp_hex);
        }
        
        // 3) Create variations based on the warp's base
        // Each thread within the warp can work on different rotation values
        int max_rotations = warp_base; // Each base gets its own number of rotations
        
        // Distribute rotations across threads in the warp
        // Each thread handles different rotations to maximize parallelism
        for (int base_rotation = lane_id; base_rotation < max_rotations && local_found == 0 && g_found == 0; 
             base_rotation += 32) {
            
            // 4) Create base-specific variation while preserving "1" prefix
            create_base_variation_preserve_prefix(&random_value, &base_variant, warp_base, base_rotation);
            
            // Convert to binary for existing transformations
            bigint_to_binary(&base_variant, binary);
            
            // Apply existing nested loop transformations
            for(int inv = 0; inv < 2 && local_found == 0 && g_found == 0; inv++) {
                for(int z = 0; z < 2 && local_found == 0 && g_found == 0; z++) {
                    for(int x = 0; x < 16 && local_found == 0 && g_found == 0; x++) {
                        
                        // Convert binary to BigInt directly - skip hex conversion
                        binary_to_bigint_direct(binary, &priv2);
                        
                        // Keep the windowed method - it's better for your use case
                        scalar_multiply_jac_device(&result_jac, &const_G_jacobian, &priv2);
                        jacobian_to_affine(&public_key, &result_jac);
                        coords_to_compressed_pubkey(public_key.x, public_key.y, pubkey);
                        hash160(pubkey, 33, hash160_out);
                        
                        local_keys_checked++;
                        
                        // Debug output for specific conditions (first thread of first warp)
                        if(warp_id == 15 && lane_id == 15 && x == 0) {
                            hash160_to_hex(hash160_out, hash160_str);
                            char hex_str[65];
                            bigint_to_hex(&priv2, hex_str);
                            printf("W%d L%d R%d: %s -> %s -> %s\n", 
                                   warp_id, lane_id, base_rotation, binary, hex_str, hash160_str);
                        }
                        
                        // Optimization 3: Early exit with minimal branching
                        if (compare_hash160_fast(hash160_out, target_bytes)) {
                            if (atomicCAS((int*)&g_found, 0, 1) == 0) {
                                // Only convert to hex when found
                                binary_to_hex(binary, temp_hex);
                                hash160_to_hex(hash160_out, hash160_str);
                                
                                memcpy(g_found_hex, temp_hex, 65);
                                memcpy(g_found_hash160, hash160_str, 41);
                                
                                printf("\n*** FOUND! ***\n");
                                printf("Warp: %d, Lane: %d, Base: %d, Rotation: %d\n", 
                                       warp_id, lane_id, warp_base, base_rotation);
                                printf("Private Key: %s\n", temp_hex);
                                printf("Hash160: %s\n", hash160_str);
                            }
                            local_found = 1;
                            goto exit_all_loops; // Break all nested loops efficiently
                        }
                        
                        binary_vertical_rotate_up(binary);
                    }
                    reverseBinaryAfterFirst1(binary);
                }
                invertBinaryAfterFirst1(binary);
            }
        }
        
        // Optional: Warp-level synchronization for cooperative work
        // __syncwarp(); // Uncomment if threads within a warp need to synchronize
        
        exit_all_loops:;
        c++;
        
        // Optional: Add periodic progress reporting per warp
        if (c % 1000 == 0 && lane_id == 0) {
            printf("Warp %d (base %d): %d iterations completed\n", warp_id, warp_base, c);
        }
    }
}
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " (required <min> <max> <target>) (optional <blocks> <threads>)" << std::endl;
        return 1;
    }
    
    try {
        init_gpu_constants();
		precompute_G_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        // Allocate device memory for 3 strings
        char *d_param1, *d_param2, *d_param3;
        
        // Get string lengths
        size_t len1 = strlen(argv[1]) + 1;
        size_t len2 = strlen(argv[2]) + 1;
        size_t len3 = strlen(argv[3]) + 1;
        
        // Allocate and copy in one operation each
        cudaMalloc(&d_param1, len1);
        cudaMemcpy(d_param1, argv[1], len1, cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_param2, len2);
        cudaMemcpy(d_param2, argv[2], len2, cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_param3, len3);
        cudaMemcpy(d_param3, argv[3], len3, cudaMemcpyHostToDevice);
        
        // Parse grid configuration
        int blocks = (argc >= 5) ? std::stoi(argv[4]) : 32;
        int threads = (argc >= 6) ? std::stoi(argv[5]) : 32;
        
        printf("Launching with %d blocks and %d threads\nTotal parallel threads: %d\n\n", 
               blocks, threads, blocks * threads);
        
        // Launch kernel
        start_optimized<<<blocks, threads>>>(d_param1, d_param2, d_param3);
        
        // Wait for completion
        cudaDeviceSynchronize();
        
        // Check if solution was found
        int found_flag;
        cudaMemcpyFromSymbol(&found_flag, g_found, sizeof(int));
        
        if (found_flag) {
            char found_hex[65];
            char found_hash160[41];
            
            // Copy results from device
            cudaMemcpyFromSymbol(found_hex, g_found_hex, 65);
            cudaMemcpyFromSymbol(found_hash160, g_found_hash160, 41);
            
            // Save to file with timestamp
            std::ofstream outfile("result.txt", std::ios::app);
            if (outfile.is_open()) {
                std::time_t now = std::time(nullptr);
                char timestamp[100];
                std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", 
                             std::localtime(&now));
                
                outfile << "[" << timestamp << "] Found: " << found_hex 
                       << " -> " << found_hash160 << std::endl;
                outfile.close();
                std::cout << "Result appended to result.txt" << std::endl;
            } else {
                std::cerr << "Unable to open file for writing" << std::endl;
            }
        }
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_param1);
            cudaFree(d_param2);
            cudaFree(d_param3);
            return 1;
        }
        
        // Clean up
        cudaFree(d_param1);
        cudaFree(d_param2);
        cudaFree(d_param3);
        
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        cudaDeviceReset();
        return 1;
    }
    
    return 0;
}