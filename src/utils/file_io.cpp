#include "file_io.h"
#include <fstream>

bool jetnet::write_binary_file(const void* data, size_t len, std::string filename)
{
    std::ofstream outfile(filename, std::ofstream::binary);

    if (!outfile)
        return false;

    // write size to file
    //outfile.write(reinterpret_cast<const char *>(len), sizeof(size_t));
    // write data to file
    outfile.write(reinterpret_cast<const char *>(data), len);
    return outfile.good();
}
