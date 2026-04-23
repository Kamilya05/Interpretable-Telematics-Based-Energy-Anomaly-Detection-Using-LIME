
# Explaining Vehicle Energy Anomalies in Telematics with XGBoost, Manual SHAP, and LIME

**A tutorial-style case study on residual anomaly detection, post-hoc explanation, and trustworthiness auditing for trip-level vehicle energy analytics**

---

## 1. Introduction

Modern vehicles produce rich telematics streams: speed, acceleration, idling, current and voltage summaries, and vehicle metadata. These signals are useful for understanding energy efficiency, but in practice a raw anomaly score is not enough. If a system flags a trip as suspicious, a driver, analyst, or engineer immediately asks a second question:

**Why did the model flag this trip?**

This blog post presents a full end-to-end case study for **vehicle energy anomaly detection with explanations**. The central goal is not only to detect trips with unexpectedly high energy consumption, but also to explain those alerts with transparent post-hoc XAI and to audit whether those explanations deserve trust.

The final lesson is simple:

> **A working XAI pipeline is not automatically a trustworthy one.**

---

## 2. Domain of application

This project lies at the intersection of:

- vehicle telematics,
- trip-level energy-efficiency analytics,
- anomaly detection,
- explainable AI,
- and operational robustness auditing.

The task is to analyze real driving data and identify trips whose energy consumption is unusually high relative to what a learned baseline model would expect.

Potential users include:

- fleet analysts,
- EV / PHEV telemetry researchers,
- automotive product teams,
- and XAI practitioners interested in structured-data explanations.

---

## 3. Motivation

There are two reasons this problem is interesting.

### 3.1 Practical motivation

A trip with high energy usage may indicate:
- heavy stop-go traffic,
- unusually aggressive driving,
- long idle periods,
- route-specific conditions,
- missing contextual information,
- or possible degradation signals.

An anomaly detector that only outputs a score is therefore hard to use. Users need **reasons**, not just alerts.

### 3.2 XAI motivation

This is a good XAI case study because the setup is realistic but still tractable:

- the model is a strong tabular black box,
- anomalies are defined by residual behavior,
- explanations can be evaluated locally,
- and trust problems such as leakage and instability are visible in practice.

---

## 4. The real-world problem

We want to identify **unexpectedly energy-inefficient trips**.

The key phrase is **unexpectedly**. We are not simply looking for trips with high absolute energy consumption. Long trips naturally consume more energy. Instead, we ask:

> *Given the trip’s telematics profile and vehicle metadata, how much energy should this trip normally consume?*

If the actual energy consumption is much higher than the model’s expectation, the trip is flagged as anomalous.

This framing is important because it turns anomaly detection into a comparison against a learned notion of “normal”.

---

## 5. Dataset and trip-level representation

We use the **Vehicle Energy Dataset (VED)** and aggregate each trip into a single row.

The notebooks were developed around the Kaggle-hosted dataset:

<https://www.kaggle.com/datasets/galievilyas/ved-dataset/data>


### 5.1 Unit of analysis

Each row corresponds to **one trip**.

### 5.2 Target

The target is:

`energy_per_km`

Using energy per kilometer is preferable to total trip energy because it normalizes for trip length and focuses the model on *efficiency* rather than scale.

### 5.3 Feature groups

The trip-level table contains features such as:

- `duration_min`, `distance_km`
- `speed_mean`, `speed_var`
- `accel_mean`, `accel_var`, `accel_p95`
- `idle_time_min`, `stop_go_ratio`
- `Generalized_Weight`
- one-hot vehicle descriptors such as transmission and drive wheels

These features summarize driving intensity, traffic pattern, vehicle load, and operating context.

---

## 6. Black-box model and anomaly score

## 6.1 Predictive model

The black-box model is an **XGBoost regressor** trained to predict expected trip-level `energy_per_km`.

Why XGBoost?

- it is a strong baseline for tabular data,
- it handles nonlinear interactions,
- it works well with heterogeneous features,
- and it integrates naturally into an explanation pipeline.

## 6.2 Residual-based anomaly formulation

Let

- `y` be the actual trip energy per kilometer,
- `ŷ` be the model prediction.

We define the residual:

`residual = y - ŷ`

We are interested only in **positive residuals**, because they correspond to trips that consumed **more energy than expected**.

A trip is anomalous if its positive residual exceeds a chosen threshold.

This is different from generic anomaly detection. We do not detect arbitrary outliers; we detect **unexpected inefficiency under the model**.

---

## 7. Mathematical formulation

Let `x_i` denote the trip-level feature vector for trip `i`, and let `f(x_i)` be the XGBoost prediction.

### 7.1 Prediction
`ŷ_i = f(x_i)`

### 7.2 Residual
`r_i = y_i - ŷ_i`

### 7.3 Anomaly rule
`trip i is anomalous if r_i > τ`

where `τ` is a threshold chosen from validation residuals.

This makes the pipeline:

1. learn expected energy consumption,
2. compare actual against expected,
3. explain the trips with the largest positive deviation.

---

## 8. Main predictive results

On the test split, the shipped baseline achieves:

| Metric | Value |
|---|---:|
| MAE | 0.0670 |
| RMSE | 0.1332 |
| R² | 0.5377 |
| Number of test trips | 698 |
| Flagged anomalies | 41 |
| Anomaly rate | 5.87% |

These numbers show that the baseline model is accurate enough to define a meaningful residual anomaly score.

---

## 9. Why explanation is needed

The anomaly score tells us **what** happened:

- this trip consumed more energy than expected.

The explanation should tell us **why** the model thinks so.

Without explanation:
- the alert is not actionable,
- failure modes remain hidden,
- and the system cannot be meaningfully audited.

In this project we compare two XAI approaches:

1. **Manual SHAP** — the primary explanation method
2. **LIME** — a local surrogate baseline

---

## 10. Manual SHAP from first principles

## 10.1 Why Manual SHAP?

Instead of relying only on the `shap` library, we implement a **manual Shapley-style explainer**. The goal is pedagogical: to make the core logic of SHAP visible.

This is **not** the optimized TreeSHAP algorithm. It is a transparent implementation of Shapley-based feature attribution.

## 10.2 Shapley value definition

For a feature `j`, the Shapley value is the average marginal contribution of that feature across all coalitions of the remaining features:

`φ_j = Σ_{S ⊆ F \ {j}} [ |S|! (M-|S|-1)! / M! ] · ( v(S ∪ {j}) - v(S) )`

where:

- `F` is the full feature set,
- `M = |F|`,
- `v(S)` is the model value when only features in coalition `S` are fixed.

## 10.3 Coalition value in our implementation

To define `v(S)` for tabular telematics data, we use **background averaging**:

- fix coalition features to their values from the trip being explained,
- fill the remaining features using background rows from training data,
- run the model on those masked background rows,
- average the predictions.

So:

`v(S) = E_background [ f(x with features in S fixed to x0) ]`

This makes the explanation interventional and model-based.

## 10.4 Exact vs approximate SHAP

Exact Shapley computation is exponential in the number of features, so we use two versions:

- **exact Shapley values** on very small feature subsets,
- **permutation-based approximate SHAP** for practical local and global explanations.

## 10.5 Permutation SHAP approximation

In the practical version:

1. start from a baseline prediction under the background distribution,
2. choose a random order of features,
3. reveal features one by one,
4. measure the prediction change at each step,
5. average these marginal contributions over many permutations.

This approximates the Shapley value efficiently enough for tutorial-scale analysis.

---

## 11. How we implemented Manual SHAP

The repository contains a `ManualSHAP` class in `06_shap_vs_lime.ipynb` that supports:

- `exact_shap_row(...)`
- `permutation_shap_row(...)`
- `explain_many(...)`
- `plot_summary(...)`
- `plot_local(...)`

### 11.1 Core implementation idea

For a single trip:

- choose a small background sample from `X_train`,
- compute the base value as the mean prediction over the background,
- repeatedly reveal features in random order,
- accumulate marginal changes.

### 11.2 Practical optimizations

To keep the implementation usable:

- the base value is cached,
- background size is capped,
- exact coalition values are memoized,
- permutation updates reuse the current masked matrix instead of rebuilding it from scratch at every step.

### 11.3 What the global SHAP summary showed

Across many trips, the strongest feature groups were:

- transmission indicators,
- acceleration variance,
- generalized weight,
- distance,
- duration.

This is a good sign: the model appears to rely on interpretable trip-level structure.

---

## 12. LIME as a comparison baseline

LIME explains a single prediction by fitting an interpretable local surrogate model. Implemented in `src/xai/`

### 12.1 LIME algorithm in our setup

For a selected trip `x0`:

1. generate perturbed neighbors around `x0`,
2. query the black-box regressor on those neighbors,
3. weight neighbors by proximity,
4. fit a weighted ridge regression surrogate,
5. interpret the top local coefficients.

### 12.2 Why we keep LIME

LIME is useful for three reasons:

- it is a classic XAI baseline,
- it gives straightforward local feature weights,
- it exposes a critical lesson: **a readable explanation is not automatically a faithful explanation**.

### 12.3 Local fidelity metrics

We evaluate LIME with:
- local `R²`,
- local RMSE,
- and stability across random seeds.

This is crucial, because LIME can produce visually clean explanations even when the surrogate is a weak approximation.

---

## 13. Local case studies

We analyze four anomalous trips in detail.

## 13.1 Trip `536_340` — strong anomaly, strong local fit

| Quantity | Value |
|---|---:|
| Actual energy per km | 1.5590 |
| Predicted energy per km | 0.4924 |
| Residual | 1.0666 |
| Transmission | NO DATA |
| LIME local R² | 0.8557 |

**Manual SHAP top contributors**
- `Transmission_CVT: 0.0896`
- `accel_var: 0.0221`
- `Transmission_NO DATA: 0.0148`

**LIME top contributors**
- `speed_var: -4.0517`
- `speed_mean: -0.1761`
- `Generalized_Weight: 0.0194`

Interpretation:
this trip is clearly anomalous, and both methods agree that it lies far outside the model’s normal operating profile, although they emphasize different factors.

## 13.2 Trip `11_3013` — moderate anomaly, stable explanation

| Quantity | Value |
|---|---:|
| Actual energy per km | 1.1895 |
| Predicted energy per km | 0.8825 |
| Residual | 0.3070 |
| Transmission | CVT |
| LIME local R² | 0.8172 |

Interpretation:
a smaller anomaly than the top outliers, but still locally explainable in a relatively stable way.

## 13.3 Trip `443_1365` — moderate anomaly, stable explanation

| Quantity | Value |
|---|---:|
| Actual energy per km | 0.7369 |
| Predicted energy per km | 0.4854 |
| Residual | 0.2515 |
| Transmission | CVT |
| LIME local R² | 0.8306 |

Interpretation:
another moderate anomaly where both explanation methods remain coherent.

## 13.4 Trip `560_141` — strongest anomaly, cautionary case

| Quantity | Value |
|---|---:|
| Actual energy per km | 2.2628 |
| Predicted energy per km | 0.6627 |
| Residual | 1.6001 |
| Transmission | CVT |
| LIME local R² | 0.1011 |

Interpretation:
this is the strongest anomaly in the test split, but LIME’s local fit is very poor. It is the best example of why explanation readability does **not** automatically imply faithfulness.

---

## 14. SHAP vs LIME: what did we learn?

The comparison between Manual SHAP and LIME is not about deciding that one is always “right” and the other is always “wrong”.

They answer slightly different questions:

- **Manual SHAP** provides additive feature attributions relative to a model-based baseline.
- **LIME** provides a local linear approximation around a specific point.

In practice:

- SHAP is better as the primary explanation layer,
- LIME is useful as a comparison baseline,
- and LIME especially helps surface trust problems through local fidelity and instability.

---

## 15. Trustworthiness audit

This is the most important part of the project.

We did not stop at “the model works” and “the explanations look nice”.  
We explicitly audited whether the whole pipeline is trustworthy.

### 15.1 Leakage ablation

We compared a **clean** model and a **leaky** model.

| Variant | n_features | val_mae | val_rmse | val_r2 | test_mae | test_rmse | test_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 20 | 0.0589 | 0.0885 | 0.7320 | 0.0667 | 0.1276 | 0.5763 |
| leaky | 24 | 0.0271 | 0.0419 | 0.9399 | 0.0415 | 0.0859 | 0.8079 |

The leaky model looks dramatically better.

But this is exactly the point: **high apparent performance can be misleading** when feature selection is not carefully audited.

So the cleaned model is the more trustworthy benchmark even though its metrics are less impressive.

### 15.2 Validation-only threshold calibration

We calibrated the anomaly threshold on validation residuals.

| Threshold | Value | Test anomaly rate | Test anomalies |
|---|---:|---:|---:|
| validation positive q95 | 0.0864 | 0.0960 | 67 |
| validation positive q98 | 0.1326 | 0.0688 | 48 |

This shows that anomaly prevalence depends strongly on threshold calibration. That is why threshold selection must be treated as part of the model pipeline, not as an afterthought.

### 15.3 LIME stability audit

We evaluated LIME across random seeds.

| trip_id | mean_jaccard_top5 | mean_sign_consistency | mean_local_r2 | mean_local_rmse |
|---|---:|---:|---:|---:|
| 536_340 | 1.0000 | 0.7250 | 0.5528 | 0.0125 |
| 11_3013 | 0.7500 | 0.6708 | 0.4702 | 0.0192 |
| 443_1365 | 0.7381 | 0.6905 | 0.5804 | 0.0147 |
| 560_141 | 0.7262 | 0.8125 | 0.5265 | 0.0033 |

This tells us that:
- the explanations are not wildly random,
- but stability is clearly case-dependent,
- and local fidelity is not uniformly high.

### 15.4 Subgroup robustness

The strongest subgroup difference appears for `Transmission`.

| group_value | n | mae | rmse | r2 | anomaly_rate | mean_residual |
|---|---:|---:|---:|---:|---:|---:|
| CVT | 636 | 0.0473 | 0.0882 | 0.6143 | 0.0330 | -0.0183 |
| NO DATA | 62 | 0.2689 | 0.3466 | -0.4561 | 0.7419 | 0.2324 |

This is not a classical demographic fairness result.  
But it is a very important **operational robustness result**: performance and anomaly frequency change dramatically when metadata quality changes.

---

## 16. Implementation walkthrough

The final project is organized as three main notebooks.

### 16.1 `05_blogpost_main_flow.ipynb`
This notebook:
- loads the baseline artifact and scored trip table,
- defines residual anomalies,
- exports the main predictive figures,
- and produces the core performance summary.

### 16.2 `06_shap_vs_lime.ipynb`
This notebook:
- implements and runs Manual SHAP,
- builds the global SHAP summary,
- exports local SHAP plots,
- exports local LIME plots,
- and creates side-by-side comparison figures.

### 16.3 `07_trustworthiness_audit.ipynb`
This notebook:
- performs the leakage ablation,
- calibrates validation-only thresholds,
- runs LIME stability checks,
- and computes subgroup robustness tables and figures.

The project also includes a Streamlit demo that lets a user inspect flagged trips interactively.

---

## 17. Streamlit demo

The Streamlit app is not the scientific core of the project, but it is useful as a deployment-style layer.

It allows a user to:
- select a trip,
- inspect actual vs predicted energy,
- inspect residual and anomaly status,
- and view a local explanation.

This turns the pipeline into a practical decision-support interface rather than only a notebook-based experiment.

---

## 18. Limitations

It is important to state clearly what this project **does not** claim.

### 18.1 Anomaly is not diagnosis
A flagged trip means the trip is unusual **under the model**, not that the vehicle definitely has a fault.

### 18.2 Post-hoc explanations are model explanations
SHAP and LIME explain the model’s behavior, not causal reality.

### 18.3 Trip aggregation loses temporal structure
Short-lived events, bursty behavior, and ordering effects are compressed into one row.

### 18.4 Missing context remains a challenge
Weather, route elevation, detailed traffic information, and sensor reliability are not fully represented.

These are real limitations, and they define the scope of our conclusions.

---

## 19. What we learned

This project taught us four main lessons.

1. **Residual-based anomaly detection is a clean and interpretable starting point** for telematics energy analytics.

2. **Manual SHAP is valuable pedagogically** because it makes the logic of feature attribution explicit.

3. **LIME is useful, but must be audited**. A nice local bar chart is not enough.

4. **Trustworthy XAI requires explicit auditing**:
   - leakage checks,
   - calibration discipline,
   - stability analysis,
   - and subgroup robustness checks.

---

## 20. Conclusions

We built a full tutorial-style workflow for:

- predicting expected trip-level energy use,
- flagging unexpectedly inefficient trips,
- explaining those alerts with Manual SHAP and LIME,
- and auditing where those explanations can and cannot be trusted.

The strongest conclusion of the project is not simply that XAI can be added to a telematics model.

It is this:

> **A high-performing, visually convincing XAI pipeline may still be misleading unless trustworthiness is audited explicitly.**

In this sense, the project is not only about energy anomalies.  
It is also about how to build more responsible XAI case studies in practice.

---

## 21. Reproducibility

Recommended notebook order:

1. `regression_model/ved-energy-regression.ipynb`
2. `notebooks/05_blogpost_main_flow.ipynb`
3. `notebooks/06_shap_vs_lime.ipynb`
4. `notebooks/07_trustworthiness_audit.ipynb`

Key output locations:

- `outputs_blogpost/figures/`
- `outputs_blogpost/xai/`
- `outputs_blogpost/audit/`
- `outputs_blogpost/tables/`

---

## 22. Final takeaway

If I had to summarize the entire project in one sentence, it would be:

> **Anomaly scores need explanations, and explanations need audits.**

## Repository Structure

```text
docs/
  blogpost_draft.md      Final narrative draft
  results_tables.md      Final metrics and interpretation tables
  reproducibility.md     Notebook order and output locations

notebooks/
  05_blogpost_main_flow.ipynb
  06_shap_vs_lime.ipynb
  07_trustworthiness_audit.ipynb
  baseline_implementation/
    final-telematics-lime.ipynb
    output.zip           Compact baseline artifact used by final notebooks

regression_model/
  ved-energy-regression.ipynb
  models/
  outputs/

src/
  anomaly/               Residual thresholding and anomaly table generation
  audit/                 Calibration, leakage, stability, subgroup checks
  cli/                   Command-line entry points
  modeling/              Feature design helpers
  utils/                 I/O, artifact, schema helpers
  viz/                   Plotting utilities
  xai/                   LIME perturbation, kernel, surrogate, fidelity code

outputs_blogpost/
  figures/               Main figures
  tables/                Summary CSVs
  audit/                 Audit figures and tables
  xai/                   Manual SHAP-style, LIME, and comparison outputs

tests/
  Unit tests for thresholds, LIME surrogate logic, CLI helpers, and audits
```

## References

1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD 2016.
2. Oh, G. S., LeBlanc, D. J., & Peng, H. (2019). Vehicle Energy Dataset (VED), A Large-scale Dataset for Vehicle Energy Consumption Research. arXiv:1905.02081.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.

## License

MIT
