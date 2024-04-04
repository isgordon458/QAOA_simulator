#include <QuEST.h>

#include <assert.h>
#include <math.h>

#include <immintrin.h>
#include "avx_mathfun.h"
#include "avx_utils.h"

#define SIMD_BIT 3
#define SIMD (1 << SIMD_BIT)

extern float H;

void rotationCompressionUnweighted(Qureg qureg, float angle, ull *graph, bool isFirstLayer)
{
    int numQubits = qureg.numQubitsInStateVec;
    ull numAmps = 1ull << qureg.numQubitsInStateVec;

    int numGates = 0;
    for (int i = 0; i < numQubits; i++)
        numGates += __builtin_popcountll(graph[i]);

    __m256i graphI[numQubits];
    for (int i = 0; i < numQubits; i++)
        graphI[i] = _mm256_set1_epi64x(graph[i]);

    const __m256i ones = {1, 1, 1, 1};
    
    // unrolled popcount
    // https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx2-lookup.cpp

    const __m256i lookup = _mm256_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,

        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4);

    const __m256i low_mask = _mm256_set1_epi8(0x0f);

#pragma omp parallel for schedule(static)
    for (ull qq = 0; qq < numAmps; qq += SIMD)
    {

        __m256i qs[2] = {{qq, qq + 1, qq + 2, qq + 3},
                         {qq + 4, qq + 5, qq + 6, qq + 7}};
        
        __m256i Cps[2] = {(__m256i)_mm256_setzero_ps(),
                          (__m256i)_mm256_setzero_ps()};

        for (int i = 0; i < numQubits; i += 1)
        {
            __m256i tmp[2];
            __m256i tmp2[2];

            __m256i g = graphI[i];

            tmp[0] = _mm256_srli_epi64(qs[0], i);
            tmp[1] = _mm256_srli_epi64(qs[1], i);

            tmp[0] = tmp[0] & ones;
            tmp[1] = tmp[1] & ones;

            tmp[0] = _mm256_sub_epi64(_mm256_setzero_si256(), tmp[0]);
            tmp[1] = _mm256_sub_epi64(_mm256_setzero_si256(), tmp[1]);

            tmp[0] = g & (qs[0] ^ tmp[0]);
            tmp[1] = g & (qs[1] ^ tmp[1]);

            // unrolled popcount
            // https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx2-lookup.cpp

            tmp2[0] = tmp[0] & low_mask;
            tmp2[1] = tmp[1] & low_mask;

            tmp2[0] = _mm256_shuffle_epi8(lookup, tmp2[0]);
            tmp2[1] = _mm256_shuffle_epi8(lookup, tmp2[1]);

            tmp[0] = _mm256_srli_epi16(tmp[0], 4);
            tmp[1] = _mm256_srli_epi16(tmp[1], 4);

            tmp[0] = tmp[0] & low_mask;
            tmp[1] = tmp[1] & low_mask;

            tmp[0] = _mm256_shuffle_epi8(lookup, tmp[0]);
            tmp[1] = _mm256_shuffle_epi8(lookup, tmp[1]);

            tmp[0] = _mm256_add_epi8(tmp[0], tmp2[0]);
            tmp[1] = _mm256_add_epi8(tmp[1], tmp2[1]);

            tmp[0] = _mm256_sad_epu8(tmp[0], _mm256_setzero_si256());
            tmp[1] = _mm256_sad_epu8(tmp[1], _mm256_setzero_si256());

            Cps[0] = Cps[0] + tmp[0];
            Cps[1] = Cps[1] + tmp[1];
        }

        {
            __m256 contracted_rzz[2];

            __m256i tmp;
            __m256i Cp = avx2_mm256x2_cvtepi64_epi32(Cps[0], Cps[1]);
            tmp = _mm256_slli_epi32(Cp, 1); // Cp[I] * 2
            tmp = _mm256_sub_epi32(tmp, _mm256_set1_epi32(numGates));

            sincos256_ps(
                _mm256_mul_ps(_mm256_set1_ps(angle / 2.0), (__m256)_mm256_cvtepi32_ps(tmp)),
                &contracted_rzz[1], // sin
                &contracted_rzz[0]  // cos
            );

            if (isFirstLayer)
            {
                __m256 HHs = _mm256_set1_ps(H);
                _mm256_store_ps(qureg.stateVec.real + qq, _mm256_mul_ps(HHs, contracted_rzz[0]));
                _mm256_store_ps(qureg.stateVec.imag + qq, _mm256_mul_ps(HHs, contracted_rzz[1]));
            }
            else
            {
                __v8sf in[2];
                in[0] = _mm256_load_ps(qureg.stateVec.real + qq);
                in[1] = _mm256_load_ps(qureg.stateVec.imag + qq);
                _mm256_store_ps(qureg.stateVec.real + qq, _mm256_mul_ps(in[0], contracted_rzz[0]) - _mm256_mul_ps(in[1], contracted_rzz[1]));
                _mm256_store_ps(qureg.stateVec.imag + qq, _mm256_mul_ps(in[0], contracted_rzz[1]) + _mm256_mul_ps(in[1], contracted_rzz[0]));
            }
        }
    }
}

void rotationCompressionWeighted(Qureg qureg, float angle, bool *graph, float *weights, bool isFirstLayer)
{
    int numQubits = qureg.numQubitsInStateVec;
    ull numAmps = 1ull << qureg.numQubitsInStateVec;

#pragma omp parallel for schedule(static) 
    for (ull qq = 0; qq < numAmps; qq += SIMD)
    {
        __v4di qs[2] = {{qq,     qq + 1, qq + 2, qq + 3},
                        {qq + 4, qq + 5, qq + 6, qq + 7}};

        __v8sf sums = _mm256_setzero_ps();

        // count number of 1
        for (int i = 0; i < numQubits; i++)
        {
            __v8si qs_shli = ((__v8si)avx2_mm256x2_cvtepi64_epi32((__m256i)(qs[0]>>i), (__m256i)(qs[1]>>i)) & (__v8si)_mm256_set1_epi32(1)) << 31;

            for (int j = i + 1; j < numQubits; j++)
            {
                int idx = i * numQubits + j;
                if (graph[idx])
                {
                    __v8si flag = qs_shli ^ (((__v8si)avx2_mm256x2_cvtepi64_epi32((__m256i)(qs[0]>>j), (__m256i)(qs[1]>>j)) & (__v8si)_mm256_set1_epi32(1)) << 31);

                    __v8sf posWeights = _mm256_set1_ps(weights[idx]);
                    __v8sf negWeights = _mm256_xor_ps(posWeights, _mm256_set1_ps(-0.0));
                    sums += _mm256_blendv_ps(negWeights, posWeights, (__m256)(flag));
                }
            }
        }

        __m256 contracted_rzz[2];

        sincos256_ps(
            _mm256_mul_ps(_mm256_set1_ps(angle / 2.0), sums),
            &contracted_rzz[1], // sin
            &contracted_rzz[0]  // cos
        );

        if (isFirstLayer)
        {
            __m256 HHs = _mm256_set1_ps(H);
            _mm256_store_ps(qureg.stateVec.real + qq, _mm256_mul_ps(HHs, contracted_rzz[0]));
            _mm256_store_ps(qureg.stateVec.imag + qq, _mm256_mul_ps(HHs, contracted_rzz[1]));
        }
        else
        {
            __v8sf in[2];
            in[0] = _mm256_load_ps(qureg.stateVec.real + qq);
            in[1] = _mm256_load_ps(qureg.stateVec.imag + qq);
            _mm256_store_ps(qureg.stateVec.real + qq, _mm256_mul_ps(in[0], contracted_rzz[0]) - _mm256_mul_ps(in[1], contracted_rzz[1]));
            _mm256_store_ps(qureg.stateVec.imag + qq, _mm256_mul_ps(in[0], contracted_rzz[1]) + _mm256_mul_ps(in[1], contracted_rzz[0]));
        }
    }
}