#include "hostlog.h"
#include "commonlog.h"
#include <assert.h>
#include <process.h>    /* _beginthread, _endthread */
#include <stdio.h>      /* FILE, fopen, fclose, ... */
#include <windows.h>    /* HANDLE, WaitForSingleObject, CloseHandle, ... */
#include <string.h>

__device__ logform* dforms = 0;
logform* hforms = 0;

struct logsession
{
    bool running;
    int nthreads;
    HANDLE threadHandle;
    FILE* file;
    FILE** files;
} session;

unsigned __stdcall thread( void* in)
{
    for( int i=0; session.running; i = (i+1)%session.nthreads)
    {
        if( *hforms[i].drain)
        {
            FILE* file = session.file ? session.file : session.files[i];
            fwrite(hforms[i].buffer, 1, *hforms[i].written, file);
            *hforms[i].drain = 0;
        }
    }
    _endthreadex( 0 );
    return 0;
}

void cudaStartLog(int blocks, int threadsPerBlock, const char* opt)
{
    cudaStartLog( dim3(blocks), dim3(threadsPerBlock), opt);
}

void cudaStartLog(const dim3& grid, const dim3& block, const char* opt)
{
    int nthreads = (grid.x*grid.y*grid.z)*(block.x*block.y*block.z);
    assert( nthreads > 0);
    size_t forms_size = sizeof(logform) * nthreads;
    logform* h_dforms = 0;

    h_dforms = (logform*)malloc( forms_size);
    hforms = (logform*)malloc( forms_size);

    if( opt && strcmp( opt, "SEPARATE") == 0)
    {
        session.files = (FILE**)malloc(sizeof(FILE*)*nthreads);
        session.file = 0;
    }
    else
    {
        session.files = 0;
        session.file = fopen("log.txt", "wb");
    }    

    for( int i=0; i<nthreads; i++)
    {
        assert( cudaHostAlloc(&hforms[i].drain, sizeof(bool), cudaHostAllocMapped) == cudaSuccess);
        assert( cudaHostGetDevicePointer(&h_dforms[i].drain, hforms[i].drain, 0) == cudaSuccess);
        
        assert( cudaHostAlloc(&hforms[i].buffer, sizeof(uint8_t)*LOG_BUFFER_SIZE, cudaHostAllocMapped) == cudaSuccess);
        assert( cudaHostGetDevicePointer(&h_dforms[i].buffer, hforms[i].buffer, 0) == cudaSuccess);

        assert( cudaHostAlloc(&hforms[i].written, sizeof(size_t), cudaHostAllocMapped) == cudaSuccess);
        assert( cudaHostGetDevicePointer(&h_dforms[i].written, hforms[i].written, 0) == cudaSuccess);
        
        if( session.files)
        {
            char tid[80];
            char filename[80] = "log";
            sprintf(tid, "%d", i);
            strcat( strcat( filename, tid), ".txt");
            session.files[i] = fopen(filename, "wb");
        }
    }

    logform* d_dforms = 0;
    assert( cudaMalloc(&d_dforms, forms_size) == cudaSuccess);
    assert( cudaMemcpy(d_dforms, h_dforms, forms_size, cudaMemcpyHostToDevice) == cudaSuccess);
    assert( cudaMemcpyToSymbol(dforms, &d_dforms, sizeof(logform*)) == cudaSuccess);

    free( h_dforms);

    session.running = 1;
    session.nthreads = nthreads;
    session.threadHandle = (HANDLE)_beginthreadex( NULL, 0, thread, NULL, 0, NULL);
    assert( session.threadHandle);
}

void cudaStopLog()
{
    // make sure thread finishes
    session.running = 0;
    WaitForSingleObject( session.threadHandle, INFINITE );
    CloseHandle( session.threadHandle);

    // deallocate device memory for forms
    logform* d_dforms = 0;
    assert( cudaMemcpyFromSymbol(&d_dforms, dforms, sizeof( logform*)) == cudaSuccess);
    assert( cudaFree( d_dforms) == cudaSuccess);
    d_dforms = 0;
    assert( cudaMemcpyToSymbol(dforms, &d_dforms, sizeof(logform*)) == cudaSuccess);

    // deallocate each form field, write last logs
    for( int i=0; i<session.nthreads; i++)
    {
        if( *hforms[i].written > 0)
        {
            FILE* file = session.file ? session.file : session.files[i];
            fwrite(hforms[i].buffer, 1, *hforms[i].written, file);
        }
        assert( cudaFreeHost( hforms[i].drain) == cudaSuccess);
        assert( cudaFreeHost( hforms[i].buffer) == cudaSuccess);
        assert( cudaFreeHost( hforms[i].written) == cudaSuccess);
        if( session.files) fclose( session.files[i]);
    }
    free( hforms);
    hforms = 0;
    if( session.files) free( session.files);
    if( session.file) fclose( session.file);
    session.file = 0;
}
