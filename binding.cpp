#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "include/neural_net.hpp"

namespace py = pybind11;

PYBIND11_MODULE(my_nn, m) {
    py::class_<NeuralNet::Perceptron<float>>(m, "Perceptron")
        .def(py::init<size_t, float>(), py::arg("input_dim"), py::arg("lr") = 0.01f)
        .def("predict", &NeuralNet::Perceptron<float>::predict)
        .def("train", [](NeuralNet::Perceptron<float> &self, const py::list &dataset, int epochs) {
            std::vector<Point<float>> cpp_dataset;
            for (py::handle item : dataset) {
                auto pt = item.cast<std::pair<std::vector<float>, int>>();
                cpp_dataset.emplace_back(pt.first, pt.second);
            }
            self.train(cpp_dataset, epochs);
        })
        .def("get_weights", &NeuralNet::Perceptron<float>::get_weights)
        .def("get_bias", &NeuralNet::Perceptron<float>::get_bias);
}