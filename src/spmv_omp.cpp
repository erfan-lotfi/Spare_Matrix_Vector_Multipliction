#include "spmv.hpp"
#include <omp.h>
#include <stdexcept>

void spmv_omp(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
    if ((int)x.size() != A.cols) {
        throw std::runtime_error("x size mismatch");
    }

    y.assign(A.rows, 0.0);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < A.rows; ++i) {
        double sum = 0.0;

        #pragma omp simd reduction(+:sum)
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            sum += A.values[j] * x[A.col_idx[j]];
        }

        y[i] = sum;
    }
}

