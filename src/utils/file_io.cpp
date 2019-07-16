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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <curl/curl.h>

//curl writefunction to be passed as a parameter
size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata) {
    std::ostringstream *stream = (std::ostringstream*)userdata;
    size_t count = size * nmemb;
    stream->write(ptr, count);
    return count;
}

//function to retrieve the image as Cv::Mat data type
cv::Mat jetnet::curl_image(const char *url)
{
    //try {
    cv::Mat image;
    int cols, rows, channels, type;

    CURL *curl;
    CURLcode res;
    std::ostringstream stream;
    curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data); // pass the writefunction
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); // pass the stream ptr when the writefunction is called
    res = curl_easy_perform(curl); // start curl

    std::string output = stream.str(); // convert the stream into a string
    curl_easy_cleanup(curl); // cleanup
    std::vector<char> data = std::vector<char>( output.begin(), output.end() ); //convert string into a vector
    cv::Mat data_mat = cv::Mat(data); // create the cv::Mat datatype from the vector

    int flags = cv::ImreadModes::IMREAD_COLOR; // N.B. for full size, IMREAD_COLOR
    cv::Mat image_unmanaged = cv::imdecode(data_mat, flags); //read an image from memory buffer

    // clone image data to a managed context (need to test overhead of copy) and free stbi memory
    image = image_unmanaged.clone();
    //stbi_image_free(data);

    return image;
}


cv::Mat jetnet::read_image(std::string filename, int expected_channels)
{
    cv::Mat image;
    int cols, rows, channels, type;

    unsigned char *data = stbi_load(filename.c_str(), &cols, &rows, &channels, expected_channels);

    // return empty image on read failure
    if (!data)
        return image;

    if (expected_channels)
        channels = expected_channels;

    switch (channels) {
        case 1: type = CV_8UC1; break;
        case 2: type = CV_8UC2; break;
        case 3: type = CV_8UC3; break;
        case 4: type = CV_8UC4; break;
        default:
            // unsupported number of channels
            stbi_image_free(data);
            return image;
    }

    cv::Mat image_unmanaged(rows, cols, type, data);

    // clone image data to a managed context (need to test overhead of copy) and free stbi memory
    image = image_unmanaged.clone();
    stbi_image_free(data);

    return image;
}

void jetnet::write_image(std::string filename, cv::Mat image)
{
    std::vector<int> compression_params;
    compression_params.push_back(1);
    compression_params.push_back(60);

    cv::imwrite(filename, image, compression_params);
}
