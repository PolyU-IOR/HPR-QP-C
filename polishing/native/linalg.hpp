#pragma once
#include "common.hpp"
#include <functional>
namespace hpr_native {
struct SVD {
    Mat U, Vt;
    Vec s;
};
SVD svd(const Mat &, bool full = false);
std::pair<Vec, Mat> eigen_symmetric(const Mat &);
Sparse diagonal(const Vec &);
Sparse select(const CSR &, const std::vector<int> &rows, const std::vector<int> &cols);
std::vector<std::vector<int>> components(const Sparse &);
Vec minres(const std::function<Vec(const Vec &)> &, const std::function<Vec(const Vec &)> &,
           const Vec &, int &iterations, int &flag);
} // namespace hpr_native
