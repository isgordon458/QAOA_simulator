#ifndef UTILS_H
#define UTILS_H

#include <immintrin.h>

// https://stackoverflow.com/questions/69408063/how-to-convert-int-64-to-int-32-with-avx-but-without-avx-512
// pack64to32

// 2x 256 -> 1x 256-bit result
__always_inline
__m256i avx2_mm256x2_cvtepi64_epi32(__m256i a, __m256i b)
{
    // grab the 32-bit low halves of 64-bit elements into one vector
   __m256 combined = _mm256_shuffle_ps(_mm256_castsi256_ps(a),
                                       _mm256_castsi256_ps(b), _MM_SHUFFLE(2,0,2,0));
    // {b3,b2, a3,a2 | b1,b0, a1,a0}  from high to low

    // re-arrange pairs of 32-bit elements with vpermpd (or vpermq if you want)
    __m256d ordered = _mm256_permute4x64_pd(_mm256_castps_pd(combined), _MM_SHUFFLE(3,1,2,0));
    return _mm256_castpd_si256(ordered);
}

#endif