# Onboard Anomaly Detection Analysis Report (Section 3.3 / Section 4)

Evaluation of the Isolation Forest thruster-health monitor consumed by the RL environment (`src.env.spacecraft_env` / `src.env.robust_spacecraft_env`). Features: `[Isp (s), solar temperature]`, sampled once per mission hour; the detector flags `Isp` degradation and thermal excursions.

## 1. Confusion Matrix & P/R/F1 (default decision thresholds)

| Method | TP | FP | TN | FN | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Isolation Forest | 213 | 22 | 1747 | 418 | 0.906 | 0.338 | 0.492 |
| One-Class SVM | 122 | 109 | 1660 | 509 | 0.528 | 0.193 | 0.283 |
| Local Outlier Factor | 101 | 159 | 1610 | 530 | 0.388 | 0.160 | 0.227 |
| Elliptic Envelope | 279 | 0 | 1769 | 352 | 1.000 | 0.442 | 0.613 |

## 2. ROC / AUC

- **Elliptic Envelope**: AUC = `0.9969`
- **Isolation Forest**: AUC = `0.9710`
- **Local Outlier Factor**: AUC = `0.4739`
- **One-Class SVM**: AUC = `0.3426`

Full TPR/FPR curves exported to `anomaly_roc_data.csv`.

## 3. Detection Threshold Analysis (Isolation Forest)

- Grid: 30 thresholds across the anomaly-score quantile range.
- Max-F1 operating point: threshold = `0.4799`, F1 = `0.895`, precision = `0.818`, recall = `0.989`, FPR = `7.86%`, detection delay = `75 h`.

## 4. Full-Mission Detection Profile

- Mission length: `11,040 h`; degradation onset at hour `1000`.
- **False positives during nominal phase**: `14`
- **First alarm**: hour `1075` (detection delay `75 h`)
- **True positives (post-onset flags)**: `2564` hours

## 5. Methodology Notes

- The labeled test set is synthesized from the same physics used by the environment (nominal `Isp = 1782 s` vs. degraded `Isp -> 1514.7 s` along the log-drift curve $Isp(h) = Isp_0 - k\,\ln(1 + h/\tau)$, plus occasional thermal excursions).
- Replace `make_labeled_dataset()` / `run_mission_detection_profile()` data with flight or lab telemetry to re-run the same analysis on measured data.
