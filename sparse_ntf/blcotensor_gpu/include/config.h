#pragma once

#define _IType unsigned long long
#define _FType double

#define STASH_SIZE 4
#define INVALID_ID	((_IType) -1)
#define WARP_SIZE 32
#define TILE_SIZE WARP_SIZE
#define STASH_SIZE 4
#define NUM_STREAMS 8 // set CUDA_DEVICE_MAX_CONNECTIONS env = 1 to 32 (default is 8)
#define BLOCK_SIZE 128 // TODO tune?
