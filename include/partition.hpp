#pragma once
#include "csr_matrix.hpp"
#include <vector>

struct RowPartition {
    std::vector<int> row_starts; // size = p + 1
};

RowPartition partition_by_nnz(const CSRMatrix& A, int p);

