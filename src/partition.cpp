#include "partition.hpp"
#include <algorithm>

RowPartition partition_by_nnz(const CSRMatrix& A, int p) {
    RowPartition part;
    part.row_starts.resize(p + 1, 0);

    part.row_starts[0] = 0;
    part.row_starts[p] = A.rows;

    long long total_nnz = A.nnz;

    for (int rank = 1; rank < p; ++rank) {
        long long target = (total_nnz * rank) / p;

        int lo = 0, hi = A.rows;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (A.row_ptr[mid] < target)
                lo = mid + 1;
            else
                hi = mid;
        }
        part.row_starts[rank] = lo;
    }

    for (int i = 1; i < p; ++i) {
        if (part.row_starts[i] < part.row_starts[i - 1]) {
            part.row_starts[i] = part.row_starts[i - 1];
        }
    }

    return part;
}

