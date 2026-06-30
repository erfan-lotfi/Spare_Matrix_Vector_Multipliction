#pragma once
#include "csr_matrix.hpp"

CSRMatrix generate_random_sparse_matrix(int rows, int cols, int nnz, unsigned int seed = 42);
VectorData generate_random_vector(int n, unsigned int seed = 123);

