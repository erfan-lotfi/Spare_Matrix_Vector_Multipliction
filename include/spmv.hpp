#pragma once
#include "csr_matrix.hpp"
#include <vector>

void spmv_serial(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y);
void spmv_omp(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y);

