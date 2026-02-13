#include <iostream>
#include <immintrin.h>


int main()
{
	uint8_t buf[32 * 3] = { 0 };

	for (int32_t i = 0; i < _countof(buf); i++ ){
		buf[i] = i + 1;
	}
	__m256i xmm[32];
	__m256i data[32];

	xmm[0] = _mm256_lddqu_si256( reinterpret_cast<__m256i*>(buf + 32 * 0) );
	xmm[1] = _mm256_lddqu_si256( reinterpret_cast<__m256i*>(buf + 32 * 1) );
	xmm[2] = _mm256_lddqu_si256( reinterpret_cast<__m256i*>(buf + 32 * 2) );

	xmm[3] = _mm256_permute4x64_epi64( xmm[0], 0x4E );
	xmm[4] = _mm256_permute4x64_epi64( xmm[1], 0x4E );
	xmm[5] = _mm256_permute4x64_epi64( xmm[2], 0x4E );

	xmm[ 6] = _mm256_blend_epi32( xmm[0], xmm[3], 0xF0 );
	xmm[ 7] = _mm256_blend_epi32( xmm[0], xmm[3], 0xC3 );
	xmm[ 8] = _mm256_blend_epi32( xmm[0], xmm[3], 0x0F );
	xmm[10] = _mm256_blend_epi32( xmm[0], xmm[4], 0x3F );
	xmm[ 9] = _mm256_blend_epi32( xmm[1], xmm[3], 0x0C );
	xmm[11] = _mm256_blend_epi32( xmm[1], xmm[4], 0xF0 );
	xmm[12] = _mm256_blend_epi32( xmm[1], xmm[4], 0xC3 );
	xmm[14] = _mm256_blend_epi32( xmm[1], xmm[5], 0x30 );
	xmm[13] = _mm256_blend_epi32( xmm[2], xmm[4], 0x0C );

	xmm[20] = _mm256_blend_epi32( xmm[9], xmm[10], 0xF0 );
	xmm[21] = _mm256_blend_epi32( xmm[9], xmm[10], 0x0F );
	xmm[22] = _mm256_blend_epi32(xmm[13], xmm[14], 0xF0);


	data[0] = _mm256_shuffle_epi8(xmm[6], _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80));
	data[1] = _mm256_shuffle_epi8(xmm[7], _mm256_setr_epi8(9,10,11,12,13,14,15,0,1,0x80,0x80,0x80,0x80,0x80,0x80,0x80,9,10,11,12,13,14,15,0,1,0x80,0x80,0x80,0x80,0x80,0x80,0x80));
	data[2] = _mm256_shuffle_epi8(xmm[8], _mm256_setr_epi8(2,3,4,5,6,7,8,9,10,0x80,0x80,0x80,0x80,0x80,0x80,0x80,2,3,4,5,6,7,8,9,10,0x80,0x80,0x80,0x80,0x80,0x80,0x80));
	data[3] = _mm256_shuffle_epi8(xmm[20], _mm256_setr_epi8(11,12,13,14,15,0,1,2,3,0x80,0x80,0x80,0x80,0x80,0x80,0x80,11,12,13,14,15,0,1,2,3,0x80,0x80,0x80,0x80,0x80,0x80,0x80));
	data[4] = _mm256_shuffle_epi8(xmm[11], _mm256_setr_epi8(4,5,6,7,8,9,10,11,12,0x80,0x80,0x80,0x80,0x80,0x80,0x80,4,5,6,7,8,9,10,11,12,0x80,0x80,0x80,0x80,0x80,0x80,0x80));
	data[5] = _mm256_shuffle_epi8(xmm[12], _mm256_setr_epi8(13,14,15,0,1,2,3,4,5,0x80,0x80,0x80,0x80,0x80,0x80,0x80,13,14,15,0,1,2,3,4,5,0x80,0x80,0x80,0x80,0x80,0x80,0x80));
	data[6] = _mm256_shuffle_epi8(xmm[21], _mm256_setr_epi8(6,7,8,9,10,11,12,13,14,0x80,0x80,0x80,0x80,0x80,0x80,0x80,6,7,8,9,10,11,12,13,14,0x80,0x80,0x80,0x80,0x80,0x80,0x80));
	data[7] = _mm256_shuffle_epi8(xmm[22], _mm256_setr_epi8(15,0,1,2,3,4,5,6,7,0x80,0x80,0x80,0x80,0x80,0x80,0x80,15,0,1,2,3,4,5,6,7,0x80,0x80,0x80,0x80,0x80,0x80,0x80));


}
