Author: Santiago Herrera

This project provides a library that allows a CUDA thread to write to a file during its execution

REQUIREMENTS
    > Windows 7 or 8
    > nmake
    > cl
    > CUDA > 4.0
    > Fermi architecture GPU with compute capability > 2.0

INSTALLATION
    > Invoke nmake to create the library
    > copy the cudalog.lib library that nmake generated to the %CUDA_LIB_PATH% folder
    > copy hostlog.h and devicelog.h to the %CUDA_INC_PATH% folder

USAGE
    > Right before making the kernel call, invoke cudaStartLog(...) (declared in hostlog.h) passing as
    parameters the Grid and Block dimensions of the kernel launch
    > When the kernel finalizes (make sure it does), invoke cudaStopLog()
    > Inside the kernel you can call log(...) to write information to the log.txt file
    > If the string "SEPARATE" is passed as the third parameter to the cudaStartLog(...) function,
    each thread will write to its own log file. This file is called logX.txt, where X is the thread
    identifier. If "SEPARATE" is not specified, a single log.txt file is created.

NOTES
    > When compiling your program, make sure you pass the -lcudalog flag to the nvcc compiler

EXAMPLE
    // this creates a file log.txt and writes 10 lines with "hello" on each line
    #include "hostlog.h"
    #include "devicelog.h"
    
    __global__ void kernel( int* d_ptr)
    {
        log("hello\r\n", 7);
    }
    
    int main()
    {
        int value = 10;
        int* d_ptr;
        cudaMalloc( &d_ptr, sizeof(int));
        cudaMemcpy( d_ptr, &value, sizeof(int), cudaMemcpyHostToDevice);

        cudaStartLog(1,10);
        kernel<<<1,10>>>( d_ptr);
        cudaDeviceSynchronize(); // waits for kernel to finish
        cudaStopLog();

        cudaFree( d_ptr);
    }
