# Review guidelines

This repository is a research analysis, not a library. It calibrates a Nagel-Schreckenberg cellular
automaton to PeMS loop-detector data and measures how modelling choices move a simulated AV capacity
gain. The scripts in `scripts/` run once, in order, on a fixed dataset (`data/raw/README.md`), and the
tables in `results/` and the numbers in `README.md` come from them.

## What to prioritise

Focus the review on anything that could make a reported result wrong or misleading:

* **Traffic-flow and statistical method.** Fundamental-diagram and capacity estimation, queue discharge,
  bottleneck detection, calibration objective, bootstrap and other uncertainty intervals, variance
  decomposition, held-out and transfer tests.
* **Consistency between code and claims.** A number or statement in `README.md` that the code or
  `results/*.csv` does not support, or a method described differently from how it is implemented.
* **Data handling that changes what enters the analysis.** Screening rules, unit conversions, joins or
  filters that silently drop, duplicate or mislabel station-days.
* **Bugs that are reachable with this project's data and parameters** (top speed 3-6 cells/step,
  integer buffer h in {0, 1, 2}, 2,000-vehicle queues, 7 weekdays of District 4 data).

When you flag one of these, please say what it would change in the results, and how large the effect
is likely to be, if you can tell.

## Lower priority

These are welcome as brief suggestions but are not the focus of this pull request:

* input validation for values the pipeline never passes (empty queues, negative or fractional
  parameters, empty files, missing inputs);
* resumability, caching and partial-run behaviour of the scripts;
* dependency versions, typing, style, naming and refactoring.

## Already settled

Earlier review rounds on this pull request covered: free-flow downstream in the bottleneck rule,
corridor bootstrap multiplicity in the error budget, the `fd_peak` density range, triangle selection by
its own prediction, the per-corridor noise correction in the variance split, and bootstrap blocks built
from retained intervals (disclosed in the README; 94% of blocks are one unbroken hour). Please do not
re-raise these unless the current code reintroduces the problem.
