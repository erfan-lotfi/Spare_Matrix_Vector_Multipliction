#include "generator.hpp"
#include <random>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <stdexcept>

static long long key_of(int r, int c, int cols) {
    return 1LL * r * cols + c;
}

CSRMatrix generate_random_sparse_matrix(int rows, int cols, int nnz, unsigned int seed) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("rows and cols must be positive");
    }
    if (nnz < 0 || nnz > rows * 1LL * cols) {
        throw std::runtime_error("invalid nnz");
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> row_dist(0, rows - 1);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    std::uniform_real_distribution<double> val_dist(-10.0, 10.0);

    std::unordered_set<long long> used;
    used.reserve(nnz * 2);

    std::vector<std::vector<std::pair<int, double>>> rows_data(rows);

    while ((int)used.size() < nnz) {
        int r = row_dist(rng);
        int c = col_dist(rng);
        long long key = key_of(r, c, cols);

        if (used.insert(key).second) {
            double val = 0.0;
            while (val == 0.0) {
                val = val_dist(rng);
            }
            rows_data[r].push_back({c, val});
        }
    }

    for (int r = 0; r < rows; ++r) {
        std::sort(rows_data[r].begin(), rows_data[r].end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
    }

    CSRMatrix A;
    A.rows = rows;
    A.cols = cols;
    A.nnz = nnz;
    A.row_ptr.resize(rows + 1, 0);

    for (int r = 0; r < rows; ++r) {
        A.row_ptr[r + 1] = A.row_ptr[r] + static_cast<int>(rows_data[r].size());
    }

    A.col_idx.resize(nnz);
    A.values.resize(nnz);

    int idx = 0;
    for (int r = 0; r < rows; ++r) {
        for (const auto& entry : rows_data[r]) {
            A.col_idx[idx] = entry.first;
            A.values[idx] = entry.second;
            ++idx;
        }
    }

    return A;
}

VectorData generate_random_vector(int n, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    VectorData x;
    x.size = n;
    x.values.resize(n);

    for (int i = 0; i < n; ++i) {
        x.values[i] = dist(rng);
    }

    return x;
}

