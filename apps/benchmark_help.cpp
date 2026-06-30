#include <iostream>

int main() {
    std::cout << "Benchmark plan:\n";
    std::cout << "1) Strong scaling:\n";
    std::cout << "   - keep rows, cols, nnz fixed\n";
    std::cout << "   - vary threads/processes: 1,2,4,8,...\n\n";

    std::cout << "2) Weak scaling:\n";
    std::cout << "   - increase problem size with threads/processes\n";
    std::cout << "   - keep density = nnz / (rows*cols) constant\n\n";

    std::cout << "3) Metrics:\n";
    std::cout << "   - total time\n";
    std::cout << "   - compute time\n";
    std::cout << "   - MPI communication time\n";
    std::cout << "   - GFLOPS\n";
    std::cout << "   - speedup\n";
    return 0;
}

‍