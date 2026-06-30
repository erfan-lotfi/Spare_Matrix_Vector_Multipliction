#include "io.hpp"
#include "spmv.hpp"
#include "timer.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <omp.h>

static void append_csv(const std::string& csv_file,
                       const std::string& mode,
                       const std::string& format,
                       int rows, int cols, int nnz,
                       int workers,
                       double total_time,
                       double compute_time,
                       double comm_time,
                       double gflops,
                       double speedup,
                       double max_err) {
    std::ofstream out(csv_file, std::ios::app);
    out << mode << ","
        << format << ","
        << rows << ","
        << cols << ","
        << nnz << ","
        << workers << ","
        << total_time << ","
        << compute_time << ","
        << comm_time << ","
        << gflops << ","
        << speedup << ","
        << max_err << "\n";
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " [text|bin] matrix_file vector_file threads output_file csv_file\n";
        return 1;
    }

    std::string format = argv[1];
    std::string matrix_file = argv[2];
    std::string vector_file = argv[3];
    int threads = std::stoi(argv[4]);
    std::string output_file = argv[5];
    std::string csv_file    = argv[6];

    CSRMatrix A;
    VectorData xdata;

    bool ok = false;
    if (format == "text") {
        ok = load_csr_text(matrix_file, A) && load_vector_text(vector_file, xdata);
    } else {
        ok = load_csr_binary(matrix_file, A) && load_vector_binary(vector_file, xdata);
    }

    if (!ok) {
        std::cerr << "Failed to load input files\n";
        return 1;
    }

    std::vector<double> y_serial, y_omp;

    Timer timer;

    timer.start();
    spmv_serial(A, xdata.values, y_serial);
    double t_serial = timer.stop();

    omp_set_num_threads(threads);

    timer.start();
    spmv_omp(A, xdata.values, y_omp);
    double t_omp = timer.stop();

    double max_err = 0.0;
    for (int i = 0; i < A.rows; ++i) {
        max_err = std::max(max_err, std::abs(y_serial[i] - y_omp[i]));
    }

    double gflops = (2.0 * A.nnz) / (t_omp * 1e9);
    double speedup = t_serial / t_omp;

    if (format == "text") save_result_text(output_file, y_omp);
    else save_result_binary(output_file, y_omp);

    std::cout << "Threads:       " << threads << "\n";
    std::cout << "Serial Time:   " << t_serial << " s\n";
    std::cout << "OMP Time:      " << t_omp << " s\n";
    std::cout << "Speedup:       " << speedup << "\n";
    std::cout << "OMP GFLOPS:    " << gflops << "\n";
    std::cout << "Max Error:     " << max_err << "\n";

    append_csv(csv_file, "openmp", format, A.rows, A.cols, A.nnz,
               threads, t_omp, t_omp, 0.0, gflops, speedup, max_err);

    return 0;
}

