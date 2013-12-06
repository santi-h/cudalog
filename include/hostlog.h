#ifndef HOSTLOG_H
#define HOSTLOG_H

#include <vector_types.h> /* dim3 */

void cudaStartLog(int = 1, int = 1, const char* = 0);
void cudaStartLog(const dim3&, const dim3&, const char* = 0);

void cudaStopLog();

#endif
