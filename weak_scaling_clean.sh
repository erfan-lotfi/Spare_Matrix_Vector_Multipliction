

sizes=(5000 7071 10000 14142 20000 28284)
nnzs=(100000 200000 400000 800000 1600000 3200000)

CSV_FILE="results_weak.csv"

rm -f $CSV_FILE

for i in "${!sizes[@]}"; do
    size=${sizes[$i]}
    nnz=${nnzs[$i]}
    workers=$((2**i)) 
    if [ $workers -eq 1 ]; then workers=1; fi
    if [ $workers -gt 8 ]; then workers=8; fi  
    
    echo ""
    echo "=========================================="
    echo "📊 workers=$workers, size=$size, nnz=$nnz"
    echo "=========================================="
    
    ./gen_matrix $size $size $nnz data/A_w${workers}.txt data/A_w${workers}.bin data/x_w${workers}
    
    echo "🔹 OpenMP با $workers ترد..."
    ./run_omp bin data/A_w${workers}.bin data/x_w${workers}.bin $workers data/y_omp_w${workers}.bin $CSV_FILE
    
    echo "🔹 MPI با $workers پردازه..."
    mpirun -np $workers ./run_mpi bin data/A_w${workers}.bin data/x_w${workers}.bin data/y_mpi_w${workers}.bin $CSV_FILE
    
    echo "✅ workers=$workers کامل شد!"
done

echo ""
echo "=========================================="
echo "✅ Weak Scaling کامل شد! فایل: $CSV_FILE"
echo "=========================================="

echo "mode,format,rows,cols,nnz,workers,total_time,compute_time,comm_time,gflops,speedup,max_error" > $CSV_FILE.tmp
cat $CSV_FILE >> $CSV_FILE.tmp
mv $CSV_FILE.tmp $CSV_FILE

python plot_weak.py $CSV_FILE --output-dir plots_weak

echo "done"