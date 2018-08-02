#ifndef JETNET_FILE_IO_H
#define JETNET_FILE_IO_H

#include <string>
#include <vector>

namespace jetnet
{
    /*
     *  data:       binary data to be written
     *  len:        number of bytes in data
     *  filename:   filename to write to
     *  returns true on success, false on failure
     */
    bool write_binary_file(const void* data, size_t len, std::string filename);

    /*
     *  filename:   filename to read from
     *  returns a char vector containing the bytes of the file. On failure of reading the vector is empty
     */
    std::vector<char> read_binary_file(std::string filename);

    /*
     *  Read a text file
     *  lines:      text lines in file
     *  filename:   name of the file to read
     *  returns true on success, false on failure
     */
    bool read_text_file(std::vector<std::string>& lines, std::string filename);

    /*
     *  Save a float tensor to a file in text format (one float text value per line)
     */
    bool save_tensor_text(const float* data, size_t size, std::string filename);
};

#endif /* JETNET_FILE_IO_H */
