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

std::vector<char> jetnet::read_binary_file(std::string filename)
{
    std::ifstream infile(filename, std::ifstream::binary | std::istream::ate);

    if (!infile)
        return std::vector<char>();

    infile.seekg(0, infile.end);
    int length = infile.tellg();
    infile.seekg(0, infile.beg);

    std::vector<char> buffer(length);
    infile.read(buffer.data(), length);
    infile.close();

    return buffer;
}

bool jetnet::read_text_file(std::vector<std::string>& names, std::string filename)
{
    std::ifstream infile(filename);

    if (!infile)
        return false;

    std::string name;
    while (std::getline(infile, name)) {
        names.push_back(name);
    }

    return true;
}

bool jetnet::save_tensor_text(const float* data, size_t size, std::string filename)
{
    size_t i;
    FILE* fp;

    //TODO: rewrite with ofstream
    fp = fopen(filename.c_str(), "w");
    if (!fp)
        return false;

    for (i=0; i<size; i++) {
        fprintf(fp, "%.8f\n", data[i]);
    }

    fclose(fp);
    return true;
}
