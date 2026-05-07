# Testing Plan

## Goal

Evaluate the five prototype items on the same benchmark problems with a simple
and consistent set of metrics.

## Hardware Split

- Items `1`, `2`, and `3`: run on `Google Colab` with `GPU`
- Items `4` and `5`: run on the local laptop with `CPU`

## Problems

- `trivial_lqr`
- `nonlinear_pendulum`

For the nonlinear pendulum, test both:

- `hard` constraint mode
- `soft` constraint mode

## Metrics

For each run, record:

- wall-clock time
- iterations
- final cost
- maximum control violation
- maximum state violation

## Fixed-Horizon Comparison

Run all supported items on the same problem and mode at one baseline horizon
length `N`, then compare:

- runtime
- convergence behavior
- final cost
- constraint violation

## Horizon Scaling

Repeat the runs for two horizon lengths to measure scaling:

- `N = 20`
- `N = 200`

For each `N`, record the same metrics:

- wall-clock time
- iterations
- final cost
- maximum control violation
- maximum state violation

## Notes

- Use the same problem definition, target state, and constraint mode across
  items when comparing results.
- Fix random seeds for sampling methods.
- Record the hardware/backend used for each run.
- GPU and CPU timings come from different machines, so runtime comparisons
  should be interpreted with that in mind.
