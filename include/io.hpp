#pragma once
#include "csr_matrix.hpp"
#include <string>

bool save_csr_text(const std::string& filename, const CSRMatrix& A);
bool load_csr_text(const std::string& filename, CSRMatrix& A);

bool save_csr_binary(const std::string& filename, const CSRMatrix& A);
bool load_csr_binary(const std::string& filename, CSRMatrix& A);

bool save_vector_text(const std::string& filename, const VectorData& x);
bool load_vector_text(const std::string& filename, VectorData& x);

bool save_vector_binary(const std::string& filename, const VectorData& x);
bool load_vector_binary(const std::string& filename, VectorData& x);

bool save_result_text(const std::string& filename, const std::vector<double>& y);
bool save_result_binary(const std::string& filename, const std::vector<double>& y);

