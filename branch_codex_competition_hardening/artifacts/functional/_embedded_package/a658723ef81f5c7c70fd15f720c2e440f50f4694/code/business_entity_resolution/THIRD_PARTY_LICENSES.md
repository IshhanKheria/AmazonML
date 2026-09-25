# Runtime license record

The competition model backend recommended by the supplied remote configuration
is LightGBM, licensed under MIT. Its trained artifact is a tabular tree ensemble
and is far below the 8-billion-parameter ceiling.

Core runtime dependencies must be recorded in the final methodology package:

| Dependency | Role | License family |
|---|---|---|
| LightGBM | Recommended final classifier | MIT |
| NumPy | Numeric arrays | BSD-3-Clause |
| pandas | Tabular processing | BSD-3-Clause |
| SciPy | Numeric utilities | BSD-3-Clause |
| scikit-learn | Baseline SGD and TF-IDF | BSD-3-Clause |
| PyArrow | Parquet I/O | Apache-2.0 |
| RapidFuzz | String similarities | MIT |

The optional pretrained embedding integration is disabled in all supplied
profiles. Do not enable it for a competition run without organizer approval.
