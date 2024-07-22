#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "svo_env/svo_vec_env.h"
#include "svo_env/utils.h"


PYBIND11_MODULE(svo_env, m) {
    pybind11::class_<svo::SvoVecEnv>(m, "SVOEnv")
        .def(pybind11::init<const std::string, const std::string, const int, const bool>())
        .def("step", &svo::SvoVecEnv::step)
        .def("reset", &svo::SvoVecEnv::reset)
        .def("env_step", &svo::SvoVecEnv::env_step)
        .def("env_visualize_features", &svo::SvoVecEnv::env_visualize_features)
        .def("setSeed", &svo::SvoVecEnv::setSeed);
    m.def("load_image_batch", &load_image_batch);
}