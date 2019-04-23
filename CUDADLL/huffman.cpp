/* 包含头文件 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "huffman.h"

void huffman_table_init(HUFCODEITEM *phc, const BYTE *huftab)
{
	int  i, j, k;
	int  symbol;
	int  code;
	BYTE hufsize[256];
	WORD  hufcode[256];
	int  tabsize;

	k = 0;
	code = 0x00;
	for (i = 0; i<MAX_HUFFMAN_CODE_LEN; i++) {
		for (j = 0; j<huftab[i]; j++) {
			hufsize[k] = i + 1;
			hufcode[k] = code;
			code++; k++;
		}
		code <<= 1;
	}
	tabsize = k;    //编码长度

	for (i = 0; i<tabsize; i++) {
		symbol = huftab[MAX_HUFFMAN_CODE_LEN + i];
		phc[symbol].depth = hufsize[i];
		phc[symbol].code = hufcode[i];
	}
}
void Newhuffman_table_init(UINT32 *phc, const BYTE *huftab, const bool is_ac) {
	int  i, j, k;
	int  symbol;
	int  code;
	BYTE    hufsize[256];
	UINT32  hufcode[256];
	int  tabsize;

	k = 0;
	code = 0x00;
	for (i = 0; i<MAX_HUFFMAN_CODE_LEN; i++) {
		for (j = 0; j<huftab[i]; j++) {
			hufsize[k] = i + 1;
			hufcode[k] = code;
			code++; k++;
		}
		code <<= 1;
	}
	tabsize = k;    //编码长度

	for (i = 0; i<tabsize; i++) {
		symbol = huftab[MAX_HUFFMAN_CODE_LEN + i];
		phc[symbol] = (hufcode[i] << (32 - hufsize[i])) | hufsize[i];
	}
	// reserve first index in GPU version of AC table for special purposes
	if (is_ac) {
		phc[256] = phc[0];
		phc[0] = 0;
	}
}