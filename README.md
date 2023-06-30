Results:
For dimensions 4096 x 4096
    Computing of iteration space: 3.4806442260742188
    Overhead completed. Time 0.7985546588897705
    Total time of looping: 0.010806560516357422
    Total time: 4.26

    Gustav time: 6.964836835861206
    np.matmul: 0.7625892162322998

Identify sections of code in the overhead and in the computation of the iteration space that can be parallelized. This
could potentially reach performance beyond np.matmul Even though the total does not look very good. Its very likely
that most of them can be computed in parallel with no problems. 