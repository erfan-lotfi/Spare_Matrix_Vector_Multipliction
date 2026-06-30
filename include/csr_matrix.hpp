#pragma once
#include <vector>

struct CSRMatrix {
    int rows = 0;
    int cols = 0;
    int nnz  = 0;

    std::vector<int> row_ptr;      // size = rows + 1
    std::vector<int> col_idx;      // size = nnz
    std::vector<double> values;    // size = nnz
};

struct VectorData {
    int size = 0;
    std::vector<double> values;
};

