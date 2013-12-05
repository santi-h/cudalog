#ifndef COMMONLOG_H
#define COMMONLOG_H

#include <stdint.h> /* uint8_t */
#include <cstddef> /* size_t */

#define LOG_BUFFER_SIZE 10

struct logform
{
    bool* drain;
    uint8_t* buffer;
    size_t* written;
};

#endif
