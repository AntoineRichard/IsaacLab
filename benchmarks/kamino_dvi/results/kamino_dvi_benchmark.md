# Kamino DVI Solver Benchmark

RSL-RL training benchmark; values are three-seed means ± two-sided 95% Student-t confidence intervals.
Steady-state runtime excludes iterations 1–10; learning metrics average the final 20 iterations.

## Key findings

- Isaac-Ant-Direct: tuned DVI is 4.9× faster than current Kamino and 5.0× faster than PR3570 P-ADMM.
- Isaac-Ant-Direct: tuned DVI remains 2.5× slower than MJWarp and remains 1.4× slower than PhysX.
- Isaac-Cartpole-Direct: tuned DVI is 2.0× faster than current Kamino and 2.0× faster than PR3570 P-ADMM.
- Isaac-Cartpole-Direct: tuned DVI remains 1.5× slower than MJWarp and is approximately equal to PhysX.
- Isaac-Fourbar-Pole-Swingup: tuned DVI is 1.9× faster than current Kamino and 1.9× faster than PR3570 P-ADMM.
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
| Isaac-Fourbar-Pole-Swingup | Kamino current | 4096 | 1.260 ± 0.016 | 52151 ± 607 | 3.283 ± 0.703 | 300.000 ± 0.000 | N/A |
| Isaac-Fourbar-Pole-Swingup | PR3570 tuned DVI | 4096 | 0.647 ± 0.029 | 102082 ± 4570 | 3.469 ± 0.438 | 300.000 ± 0.000 | N/A |
| Isaac-Fourbar-Pole-Swingup | PR3570 P-ADMM | 4096 | 1.256 ± 0.049 | 52374 ± 1841 | 3.862 ± 0.302 | 300.000 ± 0.000 | N/A |
| Isaac-Velocity-Flat-AnymalD | Kamino current | 4096 | 6.057 ± 0.183 | 16307 ± 423 | 21.668 ± 0.335 | 988.969 ± 6.014 | 0.997 ± 0.002 |
| Isaac-Velocity-Flat-AnymalD | PR3570 tuned DVI | 4096 | 0.889 ± 0.004 | 110716 ± 450 | 21.683 ± 0.509 | 977.112 ± 13.521 | 0.995 ± 0.005 |
| Isaac-Velocity-Flat-AnymalD | PR3570 P-ADMM | 4096 | 6.317 ± 0.107 | 15606 ± 254 | 21.722 ± 0.597 | 985.220 ± 3.083 | 0.997 ± 0.007 |
| Isaac-Velocity-Flat-AnymalD | MJWarp | 4096 | 0.371 ± 0.014 | 265486 ± 10653 | 15.859 ± 15.941 | 978.694 ± 31.812 | 0.750 ± 1.069 |
| Isaac-Velocity-Flat-AnymalD | PhysX | 4096 | 0.479 ± 0.012 | 205414 ± 5189 | 19.574 ± 2.938 | 986.785 ± 3.059 | 0.995 ± 0.003 |

## Incomplete cells: descriptive successful seeds

These are descriptive per-seed results from successful runs in incomplete cells. They are excluded from summaries, plots, confidence intervals, and comparative speedups.

| Task | Variant | Seed | Envs | Completed | Iteration [s] | Total FPS | Reward | Episode length | Success |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IsaacContrib-DrLegs-Walk | Kamino current | 42 | 4096 | 2/3 | 8.228 | 11981 | 247.393 | 469.982 | 0.000 |
| IsaacContrib-DrLegs-Walk | Kamino current | 44 | 4096 | 2/3 | 8.168 | 12082 | 48.200 | 467.342 | 0.000 |
| IsaacContrib-DrLegs-Walk | PR3570 P-ADMM | 42 | 4096 | 2/3 | 9.785 | 10047 | 7.735 | 24.799 | 0.065 |
| IsaacContrib-DrLegs-Walk | PR3570 P-ADMM | 44 | 4096 | 2/3 | 9.722 | 10112 | 7.690 | 24.483 | 0.065 |

## Data quality and failures

- Isaac-Ant-Direct: schema v1.1 success differs from TensorBoard in 15/15 runs and 3556/4500 points; report uses TensorBoard success.
- Isaac-Cartpole-Direct: schema v1.1 success differs from TensorBoard in 15/15 runs and 1600/4500 points; report uses TensorBoard success.
- Isaac-Fourbar-Pole-Swingup: learning.success_rate.series_per_iter and TensorBoard Metrics/success_rate are absent in 9/9 runs; this is a benchmark/task-stack bug. Report shows N/A.
- Isaac-Velocity-Flat-AnymalD: schema v1.1 success differs from TensorBoard in 0/15 runs and 0/4500 points; report uses TensorBoard success.
- IsaacContrib-DrLegs-Walk: schema v1.1 success differs from TensorBoard in 0/4 runs and 0/1200 points; report uses TensorBoard success.
- Isaac-Ant-Direct PR3570 tuned DVI has seed-sensitive weak learning: the three-seed success 95% CI half-width is 0.539; this is not a runtime or stability failure.
- Isaac-Velocity-Flat-AnymalD MJWarp has seed-sensitive weak learning: the three-seed success 95% CI half-width is 1.069; this is not a runtime or stability failure.
- Bundle workspace status: 43/58 completed full runs report versions.git_dirty=true; 58/58 expose a boolean status. This field includes untracked paths and is broader than the runner's tracked-source check; true does not by itself contradict exact-HEAD validation in current manifests.
- Legacy campaign source provenance: legacy bundles record 3 distinct commits. Runner manifests did not capture exact HEAD, and these runs did not pass the current clean-source check.
- Legacy integrity limitation: TensorBoard event hash was not recorded in 45/58 completed full-run manifests; TensorBoard event files used by those runs were not retained or hashed.
- Failed full IsaacContrib-DrLegs-Walk / kamino_current / seed 43: numerical.
- Failed full IsaacContrib-DrLegs-Walk / kamino_pr_padmm / seed 43: numerical.
- Failed preflight IsaacContrib-DrLegs-Walk / kamino_pr_dvi / seed 42: numerical.
- Isaac-Cartpole-Direct PhysX has materially lower final-window reward than current Kamino.

## Figures

![runtime](runtime.png)
![learning](learning.png)

## Protocol

- RSL-RL, 300 training iterations, seeds 42–44.
- A common environment count is selected per task from 4096 downward only after explicit capacity failures.
- Current/future runner protocol: runs are sequential on one GPU and validated against exact Newton revisions and clean IsaacLab/schema ancestry for schema v1.1.
- Reward and episode length use schema series; success rate uses the matching TensorBoard trace.
