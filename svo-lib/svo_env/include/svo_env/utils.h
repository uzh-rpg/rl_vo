#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <omp.h>


pybind11::array_t<uint8_t> load_image_batch(const std::vector<std::string>& image_paths,
                                            int num_images,
                                            int height,
                                            int width);

#endif // UTILS_H