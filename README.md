# Building an Immune System for Explainable AI

Code and results for the paper *"Building an Immune System for Explainable AI: A Five-Check Audit of Autism Severity Explanations from Brain Imaging and Clinical Data"* (Susmitha A, V. Pavithra; SRM Institute of Science and Technology).

The five-check audit tests whether a model's explanations deserve trust before they reach clinicians:

1. **Predictive validity**: permutation test plus shortcut baselines (site, missing data, age and sex, recording geometry, record-level leakage)
2. **Faithfulness**: area over the perturbation curve (AOPC) vs. random removal
3. **Stability**: input noise and retraining seeds
4. **Randomized-model sanity**: explanations from a label-permuted model
5. **Plausibility and discrimination**: agreement with independent evidence

## Contents

| File | Purpose |
| --- | --- |
| `make_features_csv.py` | Builds `features.csv` from the Kinect v2 gait dataset: links each of the 800 rows to its child, marks original recordings, groups duplicate children (MD5), adds age and sex, computes the atypical-arm-posture rule |
| `case4_gait_audit.py` | Runs the five-check audit on the gait model (Case 4) |
| `inspect_dataset.py` | Optional: inspects the dataset layout |
| `case4_results.json` | Case 4 audit, all children (100 permutations) |
| `case4_results_age5to11.json` | Case 4 audit, children aged 5-11 years |
| `abide/` | Notebooks and result tables for Cases 1-3 (ABIDE I) |

## Data

The data are not included in this repository. Download them from the original sources:

- **ABIDE I**, preprocessed: http://preprocessed-connectomes-project.org/abide/
- **Kinect v2 gait dataset** (Al-Jubouri et al., 2020): https://doi.org/10.5061/dryad.s7h44j150

## Reproducing Case 4

Requires Python 3.12.

```
pip install scikit-learn shap pandas scipy statsmodels openpyxl

python make_features_csv.py "path/to/Dataset"
python case4_gait_audit.py features.csv --n-perm 100 --out case4_results.json
python case4_gait_audit.py features.csv --n-perm 100 --age-min 5 --age-max 11 --out case4_results_age5to11.json
```

The full audit takes roughly 30-60 minutes on a laptop.

## Citation

If you use this code, please cite the paper (citation to be added on publication).

## License

MIT
