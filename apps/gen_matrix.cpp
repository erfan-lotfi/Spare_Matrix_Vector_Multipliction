#include "generator.hpp"
#include "io.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage:\n";
        std::cerr << argv[0] << " rows cols nnz matrix_text matrix_bin vector_prefix\n";
        return 1;
    }

    int rows = std::stoi(argv[1]);
    int cols = std::stoi(argv[2]);
    int nnz  = std::stoi(argv[3]);

    std::string matrix_text = argv[4];
    std::string matrix_bin  = argv[5];
    std::string vector_prefix = argv[6];

    CSRMatrix A = generate_random_sparse_matrix(rows, cols, nnz, 42);
    VectorData x = generate_random_vector(cols, 123);

    if (!save_csr_text(matrix_text, A)) {
        std::cerr << "Failed to save text matrix\n";
        return 1;
    }

    if (!save_csr_binary(matrix_bin, A)) {
        std::cerr << "Failed to save binary matrix\n";
        return 1;
    }

    if (!save_vector_text(vector_prefix + ".txt", x)) {
        std::cerr << "Failed to save text vector\n";
        return 1;
    }

    if (!save_vector_binary(vector_prefix + ".bin", x)) {
        std::cerr << "Failed to save binary vector\n";
        return 1;
    }

    double density = static_cast<double>(nnz) / (1.0 * rows * cols);
    std::cout << "Generated matrix successfully\n";
    std::cout << "rows=" << rows << ", cols=" << cols << ", nnz=" << nnz << "\n";
    std::cout << "density=" << density << "\n";

    return 0;
}

