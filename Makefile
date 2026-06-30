.CXX = g++-15
MPICXX = g++-15          # به جای mpic++ از g++ استفاده می‌کنیم
CXXFLAGS = -std=c++17 -O2 -Iinclude
MPIFLAGS = -std=c++17 -O2 -Iinclude

# مسیرهای MPI برای GCC در M1
MPI_INCLUDE = -I/opt/homebrew/include
MPI_LIB = -L/opt/homebrew/lib -lmpi

# پوشه‌ها
SRC_DIR = src
APP_DIR = apps
OBJ_DIR = obj
BIN_DIR = .

# فایل‌های شیء
OBJS = $(OBJ_DIR)/io.o $(OBJ_DIR)/generator.o $(OBJ_DIR)/spmv_serial.o $(OBJ_DIR)/spmv_omp.o $(OBJ_DIR)/partition.o

# هدف‌های اصلی
all: gen_matrix run_serial run_omp run_mpi

# ایجاد پوشه obj
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# کامپایل فایل‌های شیء (برای src)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp include/*.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# برنامه‌های قابل اجرا
gen_matrix: $(OBJ_DIR)/gen_matrix.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

run_serial: $(OBJ_DIR)/run_serial.o $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

run_omp: $(OBJ_DIR)/run_omp.o $(OBJS)
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $^

# MPI - با GCC و مسیرهای صریح
run_mpi: $(OBJ_DIR)/run_mpi.o $(OBJS)
	$(MPICXX) $(MPIFLAGS) $(MPI_INCLUDE) -o $@ $^ $(MPI_LIB)

# فایل‌های شیء مخصوص apps
$(OBJ_DIR)/gen_matrix.o: $(APP_DIR)/gen_matrix.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/run_serial.o: $(APP_DIR)/run_serial.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/run_omp.o: $(APP_DIR)/run_omp.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/run_mpi.o: $(APP_DIR)/run_mpi.cpp | $(OBJ_DIR)
	$(MPICXX) $(MPIFLAGS) $(MPI_INCLUDE) -c $< -o $@

# پاکسازی
clean:
	rm -rf $(OBJ_DIR) gen_matrix run_serial run_omp run_mpi

.PHONY: all clean