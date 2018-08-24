#include "fast_image_reader.h"
#include "stb_image.h"

cv::Mat read_image(std::string filename)
{
    cv::Mat image;
    int cols, rows, channels, type;

    unsigned char *data = stbi_load(filename.c_str(), &cols, &rows, &channels, 0);

    // return empty image on read failure
    if (!data)
        return image;

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
    cv::Mat image = image_unmanaged.clone();
    stbi_image_free(data);

    return image;
}
