# Performance Analysis

---

## Results

For dimensions 4096 x 4096:

- Computing of iteration space: 4.623  
- Overhead completed. Time 0.0155  
- Total time of looping: 0.0160  
- Total time: 0.0315  

- Gustav time: 7.471  
- np.matmul: 0.765  

---

## Parallelization Potential

Identify sections of code in the overhead and in the computation of the iteration space that can be parallelized. This could potentially reach performance beyond `np.matmul`. Even though the total does not look very good, it's very likely that most of them can be computed in parallel with no problems.

Fixed a line that was consuming 98% of time in the overhead. Changed it to np.zeros() to improve performance. With precomputed iter_space the algorithm is faster than  
np.matmul. Can we improve performance on the computation of iter_space?
