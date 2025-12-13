# Finding the minimum values in the rows of a matrix (mityaeva_d_min_v_rows_matrix_dev)

- Student: Митяева Дарья Викторовна, group 3823Б1ФИ2
- Technology: SEQ | MPI
- Variant: 17

## 1. Introduction

The task implements finding the minimum values across matrix rows using parallel MPI computations. The algorithm distributes matrix rows among processes, each process finds the minima for its assigned rows, then process 0 gathers the results and broadcasts the final vector to all processes. The input data is a vector in the format [number_of_rows, number_of_columns, matrix_elements], and the output data is a vector [number_of_rows, row_minimums]. The implementation includes both SEQ and MPI versions, which have undergone functional testing and performance testing.

## 2. Problem Statement

Given a matrix represented as a flattened vector [rows, cols, matrix_elements]. Find the minimum value in each row of the matrix.

**Input format**: [rows, cols, a₁₁, a₁₂, ..., a₁ₙ, a₂₁, ..., aₘₙ] where:
- rows, cols > 0
- Matrix elements are integers

**Output format**: [rows, min₁, min₂, ..., minₙ]

## 3. Baseline Algorithm (Sequential)

The sequential algorithm (`MinValuesInRowsSEQ`) performs:

1. **Validation**: Verifies input format has correct dimensions and size
2. **Computation**: For each row, finds the minimum element
3. **Output**: Formats result as `[rows, min₁, min₂, ..., minₙ]`

**Pseudocode**:
```python
def sequential_min_per_row(matrix_data):
    rows = matrix_data[0]
    cols = matrix_data[1]
    result = []
    
    for i in range(rows):
        row_start = 2 + i * cols
        min_val = matrix_data[row_start]
        for j in range(1, cols):
            min_val = min(min_val, matrix_data[row_start + j])
        result.append(min_val)
    
    return [rows] + result
```

**Time Complexity**: O(rows × cols)
**Space Complexity**: O(rows) for storing results

## 4. Parallelization Scheme

### MPI Implementation (`MinValuesInRowsMPI`)

**Data Distribution**:

The input matrix is distributed by rows among MPI processes. Each process receives the entire input vector but processes only its assigned subset of rows. The distribution follows a block scheme: if the total number of rows rows is not evenly divisible by the number of processes size, the first remainder processes (where remainder = rows % size) receive one extra row. Each process computes the minimum values for its local rows independently.

**Communication Pattern**:

The communication follows a master-worker pattern with process 0 acting as the master. After local computation, each worker process sends its local results to process 0 using point-to-point blocking sends (MPI_Send). Process 0 receives the results from all workers using blocking receives (MPI_Recv), combines them in the correct order, and then broadcasts the final result vector to all worker processes, again using point-to-point sends. An MPI_Barrier is used at the end to synchronize all processes. There is no overlapping of computation and communication.

**Process Roles**:

- **Rank 0**: 
  - Computes the minima for its assigned rows.
  - Gathers the local results from all other processes.
  - Combines the results into the final output vector.
  - Sends the final result to every other process.

- **Other ranks**:
  - Compute the minima for their assigned rows.
  - Send their local results to process 0.
  - Receive the final result vector from process 0.

## 5. Implementation Details

tasks/
└── mityaeva_d_min_v_rows_matrix/
    ├── common/
    │   └── common.hpp
    ├── mpi/
    │   ├── include/
    │   │   └── ops_mpi.hpp
    │   └── src/
    │       └── ops_mpi.cpp
    ├── seq/
    │   ├── include/
    │   │   └── ops_seq.hpp
    │   └── src/
    │       └── ops_seq.cpp
    ├── tests/
    │   ├── functional/
    │   │   └── main.cpp
    │   └── performance/
    │       └── main.cpp
    ├── info.json
    ├── report.md
    └── settings.json

**Key Functions**:

- `ValidationImpl()` - Validates input format and dimensions
- `PreProcessingImpl()` - Prepares data structures
- `RunImpl()` - Core computation logic (differs between SEQ and MPI)
- `PostProcessingImpl()`- Validates and finalizes output

**Assumptions**:

1. Input vector follows exact format: [rows, cols, ...matrix_elements]
2. Matrix elements are stored row-major in the vector
3. All MPI processes are initialized before task execution
4. Input values fit within int
5. Number of rows and columns are positive integers

**Memory Usage**:

- Sequential Version:
  - Input storage: `O(rows × cols)` for the entire matrix
  - Output storage: `O(rows + 1)` for row count and minima
  - Temporary storage: `O(1)` during computation

- MPI Version:
  - Each process stores complete input matrix: `O(rows × cols)`
  - Local results: `O(assigned_rows)` per process
  - Communication buffers: `O(rows)` for final result distribution

## 6. Experimental Setup

**Hardware**:
- Host: AMD Ryzen 7 8845HS, 8 cores/16 threads, 16 GB DDR4 (6400 MT/s), Windows 11
- Virtual: WSL2 with Ubuntu, 8 cores allocated, 8 GB RAM allocated
- Virtual Resources: Typically 8 CPU cores and 8 GB RAM allocated to WSL for testing

Toolchain:

| Compiler | Version     | Build Type |
| -------- | ----------- | ---------- |
| g++	   | 11.4.0	     | Release    |
| MPI	   |OpenMPI 4.1.4|	-         |

**Environment Variables**:

| Variable     | Values                  | Comment                         |
| ------------ | ----------------------- | ------------------------------- |
| PPC_NUM_PROC | [1, 2, 4, 8]            | Controls number of MPI processes|
| MPI          | Standard initialization | MPI_COMM_WORLD communicator     |

**Test Data**:
All test data is generated programmatically and embedded in the source code:
- Functional tests: Six test cases in tests/functional/main.cpp covering various matrix sizes and edge cases
- Performance test: 5000×5000 matrix (25M elements) generated in tests/performance/main.cpp using formula ((i + j) % 1000) + 1 with values in range [1, 1000]

## 7. Results and Discussion

### 7.1 Correctness

Correctness was verified using Google Test framework with six comprehensive test cases:

1. Matrices with known minimum values for each row
2. Edge cases including single-element matrices, negative values, and various dimensions
3. Comparison between SEQ and MPI implementations to ensure identical results
4. Validation of output format: [rows, min₁, min₂, ..., minₙ]

All tests pass successfully for both implementations, confirming algorithmic correctness.

### 7.2 Performance

**Execution Times**:

| Mode | Processes | Time (s) | Speedup | Efficiency |
| ---- | --------- | -------- | ------- | ---------- |
| SEQ  | 1         | 0.250    | 1.00    | N/A        |
| MPI  | 1         | 0.258    | 0.97    | 97.0%      |
| MPI  | 2         | 0.135    | 1.85    | 92.5%      |
| MPI  | 4         | 0.095    | 2.63    | 65.8%      |
| MPI  | 8         | 0.085    | 2.94    | 36.8%      |

**Analysis**:

*Scaling Behavior:*
The MPI implementation demonstrates near-linear scaling up to 2 processes (92.5% efficiency), followed by diminishing returns as process count increases. The efficiency drops to 65.8% at 4 processes and further to 36.8% at 8 processes, indicating sub-optimal strong scaling beyond the optimal 2-4 process range.

*Key Bottlenecks Identified:*

1. Data Duplication: Each MPI process maintains a complete copy of the input matrix (100MB per process for the test case), leading to memory inefficiency and limiting maximum process count.
2. Communication Overhead: The master-worker communication pattern with point-to-point operations becomes increasingly costly with higher process counts.
3. Load Imbalance: Block distribution of rows causes uneven workloads, particularly when the number of rows is not evenly divisible by the number of processes.
4. Memory Bandwidth Saturation: Concurrent access to the same matrix data by multiple processes creates memory bus contention.

*Critical Path:* The communication pattern follows a sequential flow: computation → result gathering → broadcast, with no overlap between computation and communication phases.

**Performance Comparison**:

*SEQ vs MPI Single Process:*
The single-process MPI implementation shows a 3% performance penalty compared to the sequential version, attributed to MPI initialization overhead and barrier synchronization. Both implementations share identical computational complexity O(rows × cols).

Scaling Efficiency Analysis:

| Processes | Theoretical Speedup | Achieved Speedup | Efficiency | Primary Limiting Factor |
|-----------|-------------------|------------------|------------|--------------------------|
| 2         | 2.00×             | 1.85×            | 92.5%      | Near-ideal scaling       |
| 4         | 4.00×             | 2.63×            | 65.8%      | Communication overhead   |
| 8         | 8.00×             | 2.94×            | 36.8%      | Memory bandwidth saturation |

## 8. Conclusions

**What Worked**:

1. Correct Algorithm Implementation: Both sequential and MPI versions correctly compute minimum values for each matrix row across all test cases.
2. Effective Parallelization for Moderate Scales: The MPI implementation achieves near-linear speedup (1.85× with 2 processes, 92.5% efficiency) and meaningful performance improvement (2.63× with 4 processes).
3. Robust Validation: Input validation correctly identifies malformed data, and the algorithm handles various edge cases including negative values, single-element matrices, and different matrix dimensions.
4. Code Integration: Successful integration with the testing framework, passing all six functional tests with identical results between SEQ and MPI implementations.
5. Communication Pattern: The master-worker communication model functions correctly for result gathering and distribution.

**What Didn't Work**:

1. Memory Efficiency: Each MPI process stores the complete input matrix, leading to O(P×N²) memory usage instead of O(N²/P).
2. Scalability Beyond 4 Processes: Efficiency drops significantly beyond 4 processes (36.8% at 8 processes), indicating poor strong scaling for high process counts.
3. Communication Overhead Management: The sequential communication pattern (computation → gathering → broadcasting) doesn't overlap communication with computation.
4. Load Balancing: Block distribution causes uneven workloads when the number of rows isn't divisible by the number of processes.
5. Test Coverage Gap: Code coverage for recent changes (85.71%) falls below the target threshold of 95%, indicating untested edge cases.

**Limitations**:
- Architectural Limitations:
  - Flat MPI_COMM_WORLD topology without NUMA awareness
  - No support for sparse matrix representations
  - Fixed data format requiring specific vector structure
- Performance Limitations:
  - Maximum observed speedup of 2.94× (theoretical maximum ~3.2× for this implementation)
  - Communication overhead becomes dominant beyond 4 processes
  - Memory bandwidth saturation with multiple processes accessing same data
- Implementation Limitations:
  - No dynamic load balancing or adaptive work distribution
  - Basic error handling with limited recovery mechanisms
  - Point-to-point communication instead of optimized collective operations
- Scalability Constraints:
  - Strong scaling limited to 2-4 processes for optimal efficiency
  - Memory requirements grow linearly with process count
  - No support for heterogeneous computing environments
- Usability Limitations:
  - Hardcoded test data without external file support
  - Limited configuration options for different use cases
  - No performance auto-tuning based on problem characteristics

## 9. References

1. Open MPI Documentation: https://www.open-mpi.org/doc/
2. Google Test Documentation: https://google.github.io/googletest/