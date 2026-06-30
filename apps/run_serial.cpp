#include "io.hpp"
#include "spmv.hpp"
#include "timer.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " [text|bin] matrix_file vector_file output_file csv_file\n";
        return 1;
    }

    std::string format = argv[1];
    std::string matrix_file = argv[2];
    std::string vector_file = argv[3];
    std::string output_file = argv[4];
    std::string csv_file    = argv[5];

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

    std::vector<double> y;
    Timer timer;
    timer.start();
    spmv_serial(A, xdata.values, y);
    double elapsed = timer.stop();

    double gflops = (2.0 * A.nnz) / (elapsed * 1e9);

    if (format == "text") save_result_text(output_file, y);
    else save_result_binary(output_file, y);

    std::cout << "Serial Time:   " << elapsed << " s\n";
    std::cout << "Serial GFLOPS: " << gflops << "\n";

    append_csv(csv_file, "serial", format, A.rows, A.cols, A.nnz,
               1, elapsed, elapsed, 0.0, gflops, 1.0, 0.0);

    return 0;
}

