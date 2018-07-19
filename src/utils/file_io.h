#ifndef FILE_IO_H
#define FILE_IO_H

#include <string>

namespace jetnet
{
    bool write_binary_file(const void* data, size_t len, std::string filename);
};

#endif /* FILE_IO_H */
