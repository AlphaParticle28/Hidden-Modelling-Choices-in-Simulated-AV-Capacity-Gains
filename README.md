# Calibrating a mixed-traffic cellular automaton against freeway detector data

**Status (October 2026): being rebuilt.** The earlier single-station version of this project has
been retired. This README will be replaced with the full write-up once the new analysis is
complete.

## Question

Simulation studies of automated vehicles (AVs) usually report a capacity gain:
(capacity with AVs) / (capacity with human drivers) - 1. The human side of that ratio comes from a
traffic model whose parameters are often set by hand. This project calibrates the human-driver
side of a Nagel-Schreckenberg cellular automaton (with slow-to-start) against Caltrans PeMS
loop-detector data at many freeway sites and days, attaches uncertainty to every fit, and asks how
much the simulated AV gain depends on where and when the human drivers were calibrated.

## Layout

```
nasch/            simulator, measurement, data handling and calibration code
data/raw/         PeMS downloads (not in git; see data/raw/README.md)
requirements.txt
```

Further folders (`scripts/`, `tests/`, `data/processed/`, `results/`, `figures/`) are added as the
analysis is rebuilt.

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```
