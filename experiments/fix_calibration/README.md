# FIX Parser Calibration Experiments

Experiment infrastructure for optimising the CT-DP FIX
expression-template parser configuration.

**Status:** Foundation only. Feature extractors and unit tests.
Programs A-F will be added in subsequent PRs.

## Quick Start

```bash
cmake -B build -DCTDP_BUILD_EXPERIMENTS=ON -DCTDP_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure -R OnehotExtractor
```

## What Exists Now

- `common/experiment_config.h` - shared seeds, constants, consensus data
- `common/feature_extractors.h` - three extractors (onehot/36, count/40, transition/56)
- `tests/test_extractors.cpp` - 23 unit tests

## Planned (not yet implemented)

| Program | Target | Encoding | Question |
|---------|--------|----------|----------|
| A | p50 | one-hot (36) | Does pipeline work on easy target? |
| B | p99 | one-hot (36) | How much harder is p99? |
| C | log(p99) | one-hot (36) | Does log transform help? |
| D | log(p99) | + counts (40) | Do composition features help? |
| E | log(p99) | + transitions (56) | Do adjacency features help? |
| F | exhaustive | N/A | Ground truth (1,024 configs) |
