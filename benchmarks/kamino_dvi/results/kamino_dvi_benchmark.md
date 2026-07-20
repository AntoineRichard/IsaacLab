# Kamino DVI Solver Benchmark

RSL-RL training benchmark; values are three-seed means ± two-sided 95% Student-t confidence intervals.
Steady-state runtime excludes iterations 1–10; learning metrics average the final 20 iterations.

## Key findings

- Isaac-Ant-Direct: tuned DVI is 4.9× faster than current Kamino and 5.0× faster than PR3570 P-ADMM.
- Isaac-Ant-Direct: tuned DVI remains 2.5× slower than MJWarp and remains 1.4× slower than PhysX.
- Isaac-Cartpole-Direct: tuned DVI is 2.0× faster than current Kamino and 2.0× faster than PR3570 P-ADMM.
- Isaac-Cartpole-Direct: tuned DVI remains 1.5× slower than MJWarp and is approximately equal to PhysX.
- Isaac-Velocity-Flat-AnymalD: tuned DVI is 6.8× faster than current Kamino and 7.1× faster than PR3570 P-ADMM.
- Isaac-Velocity-Flat-AnymalD: tuned DVI remains 2.4× slower than MJWarp and remains 1.9× slower than PhysX.

## Summary

| Task | Variant | Envs | Iteration time [s] | Total FPS | Reward | Episode length | Success rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Isaac-Ant-Direct | Kamino current | 4096 | 2.040 ± 0.028 | 64263 ± 890 | 7402.741 ± 1650.401 | 898.716 ± 1.169 | 1.000 ± 0.000 |
| Isaac-Ant-Direct | PR3570 tuned DVI | 4096 | 0.413 ± 0.005 | 317194 ± 3709 | 10785.473 ± 3745.488 | 845.261 ± 218.999 | 0.863 ± 0.539 |
| Isaac-Ant-Direct | PR3570 P-ADMM | 4096 | 2.048 ± 0.023 | 63990 ± 722 | 8547.088 ± 4094.777 | 897.738 ± 3.259 | 1.000 ± 0.000 |
| Isaac-Ant-Direct | MJWarp | 4096 | 0.167 ± 0.001 | 785924 ± 4857 | 8872.347 ± 33.530 | 876.501 ± 25.822 | 0.921 ± 0.075 |
| Isaac-Ant-Direct | PhysX | 4096 | 0.304 ± 0.007 | 433084 ± 9769 | 6588.875 ± 1341.016 | 845.561 ± 18.860 | 0.888 ± 0.032 |
| Isaac-Cartpole-Direct | Kamino current | 4096 | 0.261 ± 0.005 | 251502 ± 4447 | 4.942 ± 0.005 | 300.000 ± 0.000 | 1.000 ± 0.000 |
| Isaac-Cartpole-Direct | PR3570 tuned DVI | 4096 | 0.133 ± 0.001 | 494241 ± 5153 | 4.950 ± 0.014 | 300.000 ± 0.000 | 1.000 ± 0.000 |
| Isaac-Cartpole-Direct | PR3570 P-ADMM | 4096 | 0.260 ± 0.002 | 252178 ± 2015 | 4.946 ± 0.014 | 300.000 ± 0.000 | 1.000 ± 0.000 |
| Isaac-Cartpole-Direct | MJWarp | 4096 | 0.087 ± 0.001 | 749904 ± 9903 | 4.947 ± 0.009 | 299.955 ± 0.193 | 1.000 ± 0.000 |
| Isaac-Cartpole-Direct | PhysX | 4096 | 0.133 ± 0.004 | 494030 ± 12633 | -5.542 ± 0.151 | 293.953 ± 5.572 | 0.970 ± 0.009 |
| Isaac-Velocity-Flat-AnymalD | Kamino current | 4096 | 6.057 ± 0.183 | 16307 ± 423 | 21.668 ± 0.335 | 988.969 ± 6.014 | 0.997 ± 0.002 |
| Isaac-Velocity-Flat-AnymalD | PR3570 tuned DVI | 4096 | 0.889 ± 0.004 | 110716 ± 450 | 21.683 ± 0.509 | 977.112 ± 13.521 | 0.995 ± 0.005 |
| Isaac-Velocity-Flat-AnymalD | PR3570 P-ADMM | 4096 | 6.317 ± 0.107 | 15606 ± 254 | 21.722 ± 0.597 | 985.220 ± 3.083 | 0.997 ± 0.007 |
| Isaac-Velocity-Flat-AnymalD | MJWarp | 4096 | 0.371 ± 0.014 | 265486 ± 10653 | 15.859 ± 15.941 | 978.694 ± 31.812 | 0.750 ± 1.069 |
| Isaac-Velocity-Flat-AnymalD | PhysX | 4096 | 0.479 ± 0.012 | 205414 ± 5189 | 19.574 ± 2.938 | 986.785 ± 3.059 | 0.995 ± 0.003 |

## Data quality and failures

- Isaac-Ant-Direct: schema v1.1 success differs from TensorBoard in 15/15 runs and 3556/4500 points; report uses TensorBoard success.
- Isaac-Cartpole-Direct: schema v1.1 success differs from TensorBoard in 15/15 runs and 1600/4500 points; report uses TensorBoard success.
- Isaac-Velocity-Flat-AnymalD: schema v1.1 success differs from TensorBoard in 0/15 runs and 0/4500 points; report uses TensorBoard success.
- Schema validation confirms every required reward, episode-length, and success field exists; this is a value mismatch, not missing data.
- Isaac-Ant-Direct PR3570 tuned DVI has seed-sensitive weak learning: the three-seed success 95% CI half-width is 0.539; this is not a runtime or stability failure.
- Isaac-Velocity-Flat-AnymalD MJWarp has seed-sensitive weak learning: the three-seed success 95% CI half-width is 1.069; this is not a runtime or stability failure.
- Isaac-Cartpole-Direct PhysX has materially lower final-window reward than current Kamino.

## Figures

![runtime](runtime.png)
![learning](learning.png)

## Protocol

- RSL-RL, 300 training iterations, seeds 42–44.
- A common environment count is selected per task from 4096 downward only after explicit capacity failures.
- Runs are sequential on one GPU and validated against immutable IsaacLab/Newton revisions and schema v1.1.
- Reward and episode length use schema series; success rate uses the matching TensorBoard trace.
