#include <QuEST.h>

#include <assert.h>
#include <math.h>

#include <immintrin.h>
#include "avx_mathfun.h"
#include "avx_utils.h"

#define AMD_LIBM_VEC_EXPERIMENTAL
#include <amdlibm.h>
#include <amdlibm_vec.h>

#define SIMD_BIT 3
#define SIMD (1 << SIMD_BIT)

extern qreal H;

void rotationCompressionUnweighted(Qureg qureg, double angle, ull *graph, bool isFirstLayer)
{
    int numQubits = qureg.numQubitsInStateVec;
    ull numAmps = 1ull << qureg.numQubitsInStateVec;

    int numGates = 0;
    for (int i = 0; i < numQubits; i++)
        numGates += __builtin_popcountll(graph[i]);

    __m512i graphI[numQubits];
    for (int i = 0; i < numQubits; i++)
        graphI[i] = _mm512_set1_epi64(graph[i]);

    const __m512i ones = {1, 1, 1, 1, 1, 1, 1, 1};
    
#pragma omp parallel for schedule(static)
    for (ull qq = 0; qq < numAmps; qq += SIMD)
    {

        __m512i Cp = _mm512_setzero_si512();
        __v8di qs = {qq,     qq + 1, qq + 2, qq + 3, qq + 4, qq + 5, qq + 6, qq + 7};

        for (int i = 0; i < numQubits; i += 1)
        {
            __m512i g = graphI[i];
    
            __v8di vec = qs >> i;
            vec = vec & ones;
            vec = -vec;
            vec = g & (qs ^ vec);

            Cp += _mm512_popcnt_epi64(vec);
        }

        {
            __m512d contracted_rzz[2];

            __m512i tmp;
            tmp = _mm512_slli_epi64(Cp, 1); // Cp[I] * 2
            tmp = _mm512_sub_epi64(tmp, _mm512_set1_epi64(numGates));

            amd_vrd8_sincos(
                _mm512_mul_pd(_mm512_set1_pd(angle / 2.0), _mm512_cvtepi64_pd(tmp)), 
                &contracted_rzz[1], // sin
                &contracted_rzz[0]  // cos
            );

            if (isFirstLayer)
            {
                __m512d HHs = _mm512_set1_pd(H);
                _mm512_store_pd(qureg.stateVec.real + qq, _mm512_mul_pd(HHs, contracted_rzz[0]));
                _mm512_store_pd(qureg.stateVec.imag + qq, _mm512_mul_pd(HHs, contracted_rzz[1]));
            }
            else
            {
                __v8df in[2];
                in[0] = _mm512_load_pd(qureg.stateVec.real + qq);
                in[1] = _mm512_load_pd(qureg.stateVec.imag + qq);
                _mm512_store_pd(qureg.stateVec.real + qq, _mm512_mul_pd(in[0], contracted_rzz[0]) - _mm512_mul_pd(in[1], contracted_rzz[1]));
                _mm512_store_pd(qureg.stateVec.imag + qq, _mm512_mul_pd(in[0], contracted_rzz[1]) + _mm512_mul_pd(in[1], contracted_rzz[0]));
            } 
        }
    }
}

void rotationCompressionWeighted(Qureg qureg, double angle, bool *graph, double *weights, bool isFirstLayer)
{
    int numQubits = qureg.numQubitsInStateVec;
    ull numAmps = 1ull << qureg.numQubitsInStateVec;

#pragma omp parallel for schedule(static) 
    for (ull qq = 0; qq < numAmps; qq += SIMD)
    {
        __v8di qs = {qq,     qq + 1, qq + 2, qq + 3, qq + 4, qq + 5, qq + 6, qq + 7};

        __v8df sums = _mm512_setzero_pd();

        // count number of 1
        for (int i = 0; i < numQubits; i++)
        {
            __v8di qs_shli = (qs >> i);// << 63;

            for (int j = i + 1; j < numQubits; j++)
            {
                int idx = i * numQubits + j;
                if (graph[idx])
                {
                    __mmask8 flag = _mm512_cmp_epi64_mask(_mm512_setzero_si512(), ((qs_shli ^ (qs >> j)) & 1), _MM_CMPINT_NE);
                    __v8df posWeights = _mm512_set1_pd(weights[idx]);
                    __v8df negWeights = _mm512_xor_pd(posWeights, _mm512_set1_pd(-0.0));
                    sums += _mm512_mask_blend_pd(flag, negWeights, posWeights);
                }
            }
        }

        __m512d contracted_rzz[2];

        amd_vrd8_sincos(
            _mm512_mul_pd(_mm512_set1_pd(angle / 2.0), (sums)), 
            &contracted_rzz[1], // sin
            &contracted_rzz[0]  // cos
        );

        if (isFirstLayer)
        {
            __m512d HHs = _mm512_set1_pd(H);
            _mm512_store_pd(qureg.stateVec.real + qq, _mm512_mul_pd(HHs, contracted_rzz[0]));
            _mm512_store_pd(qureg.stateVec.imag + qq, _mm512_mul_pd(HHs, contracted_rzz[1]));
        }
        else
        {
            __v8df in[2];
            in[0] = _mm512_load_pd(qureg.stateVec.real + qq);
            in[1] = _mm512_load_pd(qureg.stateVec.imag + qq);
            _mm512_store_pd(qureg.stateVec.real + qq, _mm512_mul_pd(in[0], contracted_rzz[0]) - _mm512_mul_pd(in[1], contracted_rzz[1]));
            _mm512_store_pd(qureg.stateVec.imag + qq, _mm512_mul_pd(in[0], contracted_rzz[1]) + _mm512_mul_pd(in[1], contracted_rzz[0]));
        }
    }
}