#include "io.hpp"
#include <fstream>
#include <iostream>

bool save_csr_text(const std::string& filename, const CSRMatrix& A) {
    std::ofstream out(filename);
    if (!out) return false;

    out << A.rows << " " << A.cols << " " << A.nnz << "\n";

    for (int v : A.row_ptr) out << v << " ";
    out << "\n";

    for (int v : A.col_idx) out << v << " ";
    out << "\n";

    for (double v : A.values) out << v << " ";
    out << "\n";

    return true;
}

bool load_csr_text(const std::string& filename, CSRMatrix& A) {
    std::ifstream in(filename);
    if (!in) return false;

    in >> A.rows >> A.cols >> A.nnz;

    A.row_ptr.resize(A.rows + 1);
    A.col_idx.resize(A.nnz);
    A.values.resize(A.nnz);

    for (int i = 0; i < A.rows + 1; ++i) in >> A.row_ptr[i];
    for (int i = 0; i < A.nnz; ++i) in >> A.col_idx[i];
    for (int i = 0; i < A.nnz; ++i) in >> A.values[i];

    return true;
}

bool save_csr_binary(const std::string& filename, const CSRMatrix& A) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    out.write(reinterpret_cast<const char*>(&A.rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&A.cols), sizeof(int));
    out.write(reinterpret_cast<const char*>(&A.nnz), sizeof(int));

    out.write(reinterpret_cast<const char*>(A.row_ptr.data()), (A.rows + 1) * sizeof(int));
    out.write(reinterpret_cast<const char*>(A.col_idx.data()), A.nnz * sizeof(int));
    out.write(reinterpret_cast<const char*>(A.values.data()), A.nnz * sizeof(double));

    return true;
}

bool load_csr_binary(const std::string& filename, CSRMatrix& A) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    in.read(reinterpret_cast<char*>(&A.rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&A.cols), sizeof(int));
    in.read(reinterpret_cast<char*>(&A.nnz), sizeof(int));

    A.row_ptr.resize(A.rows + 1);
    A.col_idx.resize(A.nnz);
    A.values.resize(A.nnz);

    in.read(reinterpret_cast<char*>(A.row_ptr.data()), (A.rows + 1) * sizeof(int));
    in.read(reinterpret_cast<char*>(A.col_idx.data()), A.nnz * sizeof(int));
    in.read(reinterpret_cast<char*>(A.values.data()), A.nnz * sizeof(double));

    return true;
}

bool save_vector_text(const std::string& filename, const VectorData& x) {
    std::ofstream out(filename);
    if (!out) return false;

    out << x.size << "\n";
    for (double v : x.values) out << v << " ";
    out << "\n";
    return true;
}

bool load_vector_text(const std::string& filename, VectorData& x) {
    std::ifstream in(filename);
    if (!in) return false;

    in >> x.size;
    x.values.resize(x.size);
    for (int i = 0; i < x.size; ++i) in >> x.values[i];
    return true;
}

bool save_vector_binary(const std::string& filename, const VectorData& x) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    out.write(reinterpret_cast<const char*>(&x.size), sizeof(int));
    out.write(reinterpret_cast<const char*>(x.values.data()), x.size * sizeof(double));
    return true;
}

bool load_vector_binary(const std::string& filename, VectorData& x) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    in.read(reinterpret_cast<char*>(&x.size), sizeof(int));
    x.values.resize(x.size);
    in.read(reinterpret_cast<char*>(x.values.data()), x.size * sizeof(double));
    return true;
}

bool save_result_text(const std::string& filename, const std::vector<double>& y) {
    std::ofstream out(filename);
    if (!out) return false;

    out << y.size() << "\n";
    for (double v : y) out << v << " ";
    out << "\n";
    return true;
}

bool save_result_binary(const std::string& filename, const std::vector<double>& y) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    int n = static_cast<int>(y.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof(int));
    out.write(reinterpret_cast<const char*>(y.data()), n * sizeof(double));
    return true;
}

