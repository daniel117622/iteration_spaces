# Performance Analysis

## Results

For dimensions 4096 x 4096:
- Computing of iteration space: 4.623522996902466
- Overhead completed. Time 0.01555013656616211
- Total time of looping: 0.01600193977355957
- Total time: 0.03155207633972168
- 
- Gustav time: 7.4716832637786865
- np.matmul: 0.765570878982544
## Parallelization Potential

Identify sections of code in the overhead and in the computation of the iteration space that can be parallelized. This could potentially reach performance beyond `np.matmul`. Even though the total does not look very good, it's very likely that most of them can be computed in parallel with no problems.

Fixed a line that was consuming 98% of time in the overhead. Changed it to np.zeros() to improve performance. With precomputed iter_space the algorithm is faster than 
np.matmul. Can we improve performance on the computation of iter_space? 