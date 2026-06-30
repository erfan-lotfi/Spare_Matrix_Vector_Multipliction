#include "csr_matrix.hpp"
#include "io.hpp"
#include "partition.hpp"
#include "spmv.hpp"
#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " [text|bin] matrix_file vector_file output_file csv_file\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string format = argv[1];
    std::string matrix_file = argv[2];
    std::string vector_file = argv[3];
    std::string output_file = argv[4];
    std::string csv_file    = argv[5];

    CSRMatrix A;
    VectorData xdata;
    std::vector<double> y_serial;

    RowPartition part;
    std::vector<int> row_starts;

    double comm_time = 0.0;
    double comp_time = 0.0;
    double total_start = MPI_Wtime();

    if (rank == 0) {
        bool ok = false;
        if (format == "text") {
            ok = load_csr_text(matrix_file, A) && load_vector_text(vector_file, xdata);
        } else {
            ok = load_csr_binary(matrix_file, A) && load_vector_binary(vector_file, xdata);
        }

        if (!ok) {
            std::cerr << "Failed to load input files\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        part = partition_by_nnz(A, size);
        row_starts = part.row_starts;
    }

    int rows = 0, cols = 0, nnz = 0, xsize = 0;

    if (rank == 0) {
        rows = A.rows;
        cols = A.cols;
        nnz = A.nnz;
        xsize = xdata.size;
    }

    double t0 = MPI_Wtime();
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - t0;

    if (rank != 0) {
        xdata.size = xsize;
        xdata.values.resize(xsize);
        row_starts.resize(size + 1);
    }

    t0 = MPI_Wtime();
    MPI_Bcast(xdata.values.data(), xsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_starts.data(), size + 1, MPI_INT, 0, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - t0;

    int r0 = row_starts[rank];
    int r1 = row_starts[rank + 1];
    int local_rows = r1 - r0;

    std::vector<int> local_row_ptr(local_rows + 1);
    std::vector<int> local_col_idx;
    std::vector<double> local_values;

    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            int rs = row_starts[p];
            int re = row_starts[p + 1];
            int local_nnz = A.row_ptr[re] - A.row_ptr[rs];

            std::vector<int> tmp_row_ptr(re - rs + 1);
            for (int i = 0; i <= re - rs; ++i) {
                tmp_row_ptr[i] = A.row_ptr[rs + i] - A.row_ptr[rs];
            }

            if (p == 0) {
                local_row_ptr = tmp_row_ptr;
                local_col_idx.assign(A.col_idx.begin() + A.row_ptr[rs],
                                     A.col_idx.begin() + A.row_ptr[re]);
                local_values.assign(A.values.begin() + A.row_ptr[rs],
                                    A.values.begin() + A.row_ptr[re]);
            } else {
                double ts = MPI_Wtime();
                MPI_Send(&local_nnz, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(tmp_row_ptr.data(), re - rs + 1, MPI_INT, p, 1, MPI_COMM_WORLD);
                MPI_Send(A.col_idx.data() + A.row_ptr[rs], local_nnz, MPI_INT, p, 2, MPI_COMM_WORLD);
                MPI_Send(A.values.data() + A.row_ptr[rs], local_nnz, MPI_DOUBLE, p, 3, MPI_COMM_WORLD);
                comm_time += MPI_Wtime() - ts;
            }
        }
    } else {
        int local_nnz = 0;

        double tr = MPI_Wtime();
        MPI_Recv(&local_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_col_idx.resize(local_nnz);
        local_values.resize(local_nnz);

        MPI_Recv(local_row_ptr.data(), local_rows + 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_col_idx.data(), local_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_values.data(), local_nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        comm_time += MPI_Wtime() - tr;
    }

    std::vector<double> y_local(local_rows, 0.0);

    double tc = MPI_Wtime();
    for (int i = 0; i < local_rows; ++i) {
        double sum = 0.0;
        for (int j = local_row_ptr[i]; j < local_row_ptr[i + 1]; ++j) {
            sum += local_values[j] * xdata.values[local_col_idx[j]];
        }
        y_local[i] = sum;
    }
    comp_time += MPI_Wtime() - tc;

    std::vector<int> recvcounts, displs;
    std::vector<double> y_global;

    if (rank == 0) {
        recvcounts.resize(size);
        displs.resize(size);
        for (int p = 0; p < size; ++p) {
            recvcounts[p] = row_starts[p + 1] - row_starts[p];
            displs[p] = row_starts[p];
        }
        y_global.resize(rows);
    }

    t0 = MPI_Wtime();
    MPI_Gatherv(y_local.data(), local_rows, MPI_DOUBLE,
                rank == 0 ? y_global.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - t0;

    double local_total = MPI_Wtime() - total_start;

    double max_total = 0.0, max_comp = 0.0, max_comm = 0.0;
    MPI_Reduce(&local_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        spmv_serial(A, xdata.values, y_serial);

        double max_err = 0.0;
        for (int i = 0; i < rows; ++i) {
            max_err = std::max(max_err, std::abs(y_serial[i] - y_global[i]));
        }

        double serial_time;
        {
            double t1 = MPI_Wtime();
            std::vector<double> tmp;
            spmv_serial(A, xdata.values, tmp);
            double t2 = MPI_Wtime();
            serial_time = t2 - t1;
        }

        double speedup = serial_time / max_total;
        double gflops = (2.0 * nnz) / (max_total * 1e9);

        if (format == "text") save_result_text(output_file, y_global);
        else save_result_binary(output_file, y_global);

        std::cout << "MPI Processes: " << size << "\n";
        std::cout << "Total Time:    " << max_total << " s\n";
        std::cout << "Compute Time:  " << max_comp << " s\n";
        std::cout << "Comm Time:     " << max_comm << " s\n";
        std::cout << "Speedup:       " << speedup << "\n";
        std::cout << "GFLOPS:        " << gflops << "\n";
        std::cout << "Max Error:     " << max_err << "\n";

        append_csv(csv_file, "mpi", format, rows, cols, nnz,
                   size, max_total, max_comp, max_comm, gflops, speedup, max_err);
    }

    MPI_Finalize();
    return 0;
}

