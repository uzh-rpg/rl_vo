#pragma once

#include <Eigen/Eigen>

// ------------ General Stuff-------------

// Define the scalar type used.
using Scalar = double;  // numpy float32
// ------------ Eigen Stuff-------------

// Using shorthand for `Vector<rows>` with scalar type.
template<int rows = Eigen::Dynamic>
using Vector = Eigen::Matrix<Scalar, rows, 1>;

// Using shorthand for `Matrix<rows, cols>` with scalar type.
template<int rows = Eigen::Dynamic, int cols = Eigen::Dynamic>
using Matrix = Eigen::Matrix<Scalar, rows, cols>;

// Using shorthand for `Matrix<rows, cols>` with scalar type.
template<int rows = Eigen::Dynamic, int cols = Eigen::Dynamic>
using MatrixRowMajor = Eigen::Matrix<Scalar, rows, cols, Eigen::RowMajor>;

// Using `Ref` for modifier references.
template<class Derived>
using Ref = Eigen::Ref<Derived>;
