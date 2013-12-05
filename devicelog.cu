#include "devicelog.h"
#include "commonlog.h"
#include <assert.h>
#include <stdint.h> /* uint8_t */

extern __device__ logform* dforms;

__device__ int getIdx()
{
    int idx = 0;
    int blocksize = blockDim.x*blockDim.y*blockDim.z;
    idx += (gridDim.x*gridDim.y*blocksize)*blockIdx.z;
    idx += (gridDim.x*blocksize)*blockIdx.y;
    idx += (blocksize)*blockIdx.x;

    idx += (blockDim.x*blockDim.y)*threadIdx.z;
    idx += (blockDim.x)*threadIdx.y;
    idx += (1)*threadIdx.x;

    return idx;
}

__device__ void log( const void* l, size_t size)
{
    assert( dforms);
    const uint8_t* log = (const uint8_t*)l;
    int tidx = getIdx();
    volatile bool* v_drain = dforms[tidx].drain;
    volatile uint8_t* v_buffer = dforms[tidx].buffer;
    volatile size_t* v_written = dforms[tidx].written;

    size_t written = *v_written;

    while( size > 0)
    {
        while( written < LOG_BUFFER_SIZE && size > 0) {
            v_buffer[written++] = *(log++);
            size--;
        }

        if( size > 0)
        {
            *v_written = written;
            *v_drain = 1;
            while( *v_drain);
            written = 0;
        }
    }

    *v_written = written;
}

/*
#include <stdio.h>
void printit()
{
    int* d_arr = 0;
    assert( cudaMemcpyFromSymbol(&d_arr, i, sizeof(int*)) == cudaSuccess);

    int* h_arr = (int*)malloc(sizeof(int)*16);
    assert(cudaMemcpy(h_arr, d_arr, sizeof(int)*16, cudaMemcpyDeviceToHost) == cudaSuccess);
    for( int i=0; i<16; i++) printf("%d\n", h_arr[i]);

    printf("hola\n");
}
//*/
