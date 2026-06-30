rm -rf data data_weak results results_weak plots plots_weak
rm -f *.csv *.tmp *.bak
rm -rf obj
rm -f gen_matrix run_serial run_omp run_mpi benchmark_help


make clean && make

mkdir -p data results plots

echo "=== Strong Scaling ==="
./gen_matrix 20000 20000 1000000 data/A.txt data/A.bin data/x
./run_serial bin data/A.bin data/x.bin data/y_serial.bin results/results.csv
for t in 1 2 4 8; do
    ./run_omp bin data/A.bin data/x.bin $t data/y_omp.bin results/results.csv
done
for p in 1 2 4 8; do
    mpirun -np $p ./run_mpi bin data/A.bin data/x.bin data/y_mpi.bin results/results.csv
done

echo "=== Weak Scaling ==="
./gen_matrix 10000 10000 400000 data/A_w1.txt data/A_w1.bin data/x_w1
./gen_matrix 14142 14142 800000 data/A_w2.txt data/A_w2.bin data/x_w2
./gen_matrix 20000 20000 1600000 data/A_w4.txt data/A_w4.bin data/x_w4
./gen_matrix 28284 28284 3200000 data/A_w8.txt data/A_w8.bin data/x_w8

for i in 1 2 4 8; do
    ./run_omp bin data/A_w${i}.bin data/x_w${i}.bin ${i} data/y_omp_w.bin results/results.csv
    mpirun -np ${i} ./run_mpi bin data/A_w${i}.bin data/x_w${i}.bin data/y_mpi_w.bin results/results.csv
done

echo "mode,format,rows,cols,nnz,workers,total_time,compute_time,comm_time,gflops,speedup,max_error" > results/results.csv.tmp
cat results/results.csv >> results/results.csv.tmp
mv results/results.csv.tmp results/results.csv

source venv/bin/activate
python plot_results.py results/results.csv --output-dir plots

echo "done"