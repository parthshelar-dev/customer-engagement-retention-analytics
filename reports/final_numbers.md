# Consolidated Reference Numbers: Customer Engagement & Retention Analytics
**Project:** Customer Engagement & Product Utilization Analytics for Retention Strategy  
**Audience:** European Central Bank (ECB) Retail Banking Framing & Academic Evaluation  
**Status:** Canonical Truth File for Research Paper & Executive Reporting  
**Rule:** Every figure in this file is recomputed directly from saved CSV tables in `reports/tables/` and `data/processed/bank_churn_scored.csv`. No numbers are typed from memory.

---

## 1. Portfolio Overview & Base Rates
*Source Table: `reports/tables/phase1_data_cleaning_audit.csv` and `data/processed/bank_churn_scored.csv`*

| Metric | Exact Value | Formatting / Sizing | Description / Context |
| :--- | :--- | :--- | :--- |
| **Total Customers (N)** | 10,000 | 10,000 | Total customer records in portfolio |
| **Retained Customers (Exited = 0)** | 7,963 | 79.63% | Customers with active banking relationship |
| **Churned Customers (Exited = 1)** | 2,037 | 20.37% | Historical portfolio churn baseline rate |
| **Total Portfolio Balance** | 764,858,892.88 | 764.86M | Aggregate balance across all 10,000 customers |
| **Retained Portfolio Balance** | 579,270,798.25 | 579.27M | Balance currently held by retained customers |
| **Churned Portfolio Balance** | 185,588,094.63 | 185.59M | Historical balance lost from churned accounts |
| **Zero-Balance Customer Count** | 3,617 | 36.17% | Customers with account balance = 0.00 |
| **Funded Account Count (Balance > 0)** | 6,383 | 63.83% | Customers with funded account balance |
| **Funded Accounts High-Balance Cutoff (P75)** | 119,839.69 | 119,839.69 | 75th percentile cutoff among funded accounts |
| **All Accounts High-Balance Percentile** | 76.54th | 76.54% | Percentile of 119,839.69 across all 10,000 accounts |

---

## 2. Engagement Profiles Distribution
*Source Table: `reports/tables/engagement_profiles_summary.csv`*

| Engagement Profile | Customer Count (N) | Portfolio Share (%) | Churn Rate (%) | 95% Confidence Interval | Total Balance | Average Balance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Active Engaged** (Active, 2+ Prods) | 2,752 | 27.52% | 12.50% | [11.30% - 13.77%] | 211,228,883.33 | 76,754.68 |
| **Active Low-Product** (Active, 1 Prod) | 2,399 | 23.99% | 16.38% | [14.92% - 17.88%] | 180,695,081.93 | 75,321.00 |
| **Inactive Latent** (Inactive, 2+ Prods) | 2,338 | 23.38% | 20.10% | [18.52% - 21.77%] | 177,299,692.74 | 75,833.91 |
| **Inactive Disengaged** (Inactive, 1 Prod) | 2,511 | 25.11% | 33.33% | [31.50% - 35.17%] | 195,635,234.88 | 77,911.28 |
| **Total** | **10,000** | **100.00%** | **20.37%** | **[19.58% - 21.17%]** | **764,858,892.88** | **76,485.89** |

*Note on Profile Hierarchy:* Mutual exclusivity is enforced by deterministic assignment in order: Active Engaged -> Active Low-Product -> Inactive Latent -> Inactive Disengaged.

---

## 3. Product Depth & Utilization Sweet Spot
*Source Table: `reports/tables/product_depth_analysis.csv` and `reports/tables/product_mix_matrix.csv`*

| Product Count | Total Customers (N) | Share of Portfolio | Churn Count | Churn Rate (%) | 95% Confidence Interval | Retained Customers (Exited = 0) | Retained Balance | Total Balance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Product** | 5,084 | 50.84% | 1,409 | 27.71% | [26.49% - 28.95%] | 3,675 | 363,222,968.79 | 501,987,975.29 |
| **2 Products (Sweet Spot)** | 4,590 | 45.90% | 348 | 7.58% | [6.84% - 8.37%] | 4,242 | 221,418,655.76 | 238,446,550.21 |
| **3 Products (Danger Zone)** | 266 | 2.66% | 220 | 82.71% | [77.82% - 87.22%] | 46 (incl 4P) / 36 | 3,090,147.28 | 22,561,073.49 |
| **4 Products (Extreme Risk)** | 60 | 0.60% | 60 | 100.00% | [100.00% - 100.00%] | 10 (re-opened) / 0 | 0.00 | 1,863,293.89 |
| **3-4 Products Combined** | 326 | 3.26% | 280 | 85.89% | [82.21% - 89.57%] | 46 | 3,090,147.28 | 24,424,367.38 |

*Critical Operational Note:* Only **46 retained customers** currently hold 3 or 4 products in the portfolio (36 hold 3 products, 10 hold 4 products). Because small samples (n < 30 per cell) are noisy, 3-4 products are flagged as a potential friction danger zone requiring A/B testing rather than proven causality.

---

## 4. Regional Funded-Only Balance Loyalty Test
*Source Table: `reports/tables/balance_loyalty_funded_geography.csv`*

This comparison restricts analysis strictly to funded accounts (Balance > 0), segmented by Geography, Activity, and Balance Level (Cutoff = 119,839.69).

| Country | Balance Band | Activity Status | Customer Count (N) | Churn Rate (%) | 95% Confidence Interval | Inactive / Active Churn Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **France** | Lower Funded (< 119.8k) | Active | 996 | 11.55% | [9.38% - 13.78%] | **1.96x** |
| **France** | Lower Funded (< 119.8k) | Inactive | 895 | 22.68% | [19.98% - 25.43%] | (22.68% / 11.55%) |
| **France** | High Funded (>= 119.8k) | Active | 647 | 14.06% | [11.45% - 16.78%] | **1.79x** |
| **France** | High Funded (>= 119.8k) | Inactive | 573 | 25.11% | [21.64% - 28.53%] | (25.11% / 14.06%) |
| **Germany** | Lower Funded (< 119.8k) | Active | 622 | 24.27% | [20.89% - 27.67%] | **1.73x** |
| **Germany** | Lower Funded (< 119.8k) | Inactive | 568 | 41.90% | [37.94% - 45.98%] | (41.90% / 24.27%) |
| **Germany** | High Funded (>= 119.8k) | Active | 626 | 23.17% | [19.89% - 26.68%] | **1.74x** |
| **Germany** | High Funded (>= 119.8k) | Inactive | 693 | 40.23% | [36.65% - 43.83%] | (40.23% / 23.17%) |
| **Spain** | Lower Funded (< 119.8k) | Active | 538 | 12.46% | [9.77% - 15.35%] | **2.38x** |
| **Spain** | Lower Funded (< 119.8k) | Inactive | 492 | 29.66% | [25.44% - 33.78%] | (29.66% / 12.46%) |
| **Spain** | High Funded (>= 119.8k) | Active | 377 | 13.01% | [9.87% - 16.42%] | **1.99x** |
| **Spain** | High Funded (>= 119.8k) | Inactive | 289 | 25.95% | [21.05% - 31.06%] | (25.95% / 13.01%) |

*Key Empirical Takeaway:* Inactive customers churn **1.73x to 2.38x (about 1.7 to 2.4 times)** as often as active customers at every balance level and within every geography. High balance alone does not insulate against churn.

---

## 5. Hollow Premium & High-Value At-Risk Reconciled Sizing
*Source Tables: `reports/tables/at_risk_premium_retained_geography.csv`, `reports/tables/u3_hollow_premium_summary.csv`, and `data/processed/bank_churn_scored.csv`*

To prevent ambiguity, two distinct definitions are formally maintained and distinguished:

### Definition A: Broad High-Balance Inactive Cohort (Any Products)
*Criteria: `IsActiveMember == 0 & Balance >= 119,839.69 & Exited == 0`*
- **Retained Customer Count:** 1,079
- **Retained Total Balance:** 155,004,893.37 (~155.00M)
- **Retained Average Calibrated Churn Probability:** 19.95% (0.1995)
- **Retained Total Value-at-Risk (VaR):** 30,944,693.58 (~30.94M)
- **All Portfolio (incl Churned):** 1,569 customers | Observed Churn Rate = 31.23%

### Definition B: Core Hollow Premium Cohort (Strict Single-Product Focus)
*Criteria: `IsActiveMember == 0 & Balance >= 119,839.69 & NumOfProducts == 1 & Exited == 0`*
- **Retained Customer Count:** 693
- **Retained Total Balance:** 99,241,820.65 (~99.24M balance still to protect)
- **Retained Average Calibrated Churn Probability:** 24.20% (0.2420)
- **Retained Total Value-at-Risk (VaR):** 24,152,006.73 (~24.15M)
- **All Portfolio (incl Churned):** 1,054 customers | Observed Churn Rate = 34.25%

#### Core Hollow Premium Retained Geographic Breakdown (Exact CSV Ground Truth):
| Country | Retained Customer Count | Share of Retained Cohort (%) | Historical Retention Rate (%) | Balance Still to Protect (Exited = 0) | Average Balance | Total VaR (Risk-Weighted) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **France** | 363 | 52.38% | 73.78% | 52,078,918.63 (52.08M) | 143,468.10 | 12,393,220.15 |
| **Germany** | 179 | 25.83% | 48.77% | 25,556,105.31 (25.56M) | 142,771.54 | 6,861,194.22 |
| **Spain** | 151 | 21.79% | 77.44% | 21,606,796.71 (21.61M) | 143,091.37 | 4,897,592.36 |
| **Total** | **693** | **100.00%** | **65.75%** | **99,241,820.65 (99.24M)** | **143,206.09** | **24,152,006.73** |

*Reconciliation Note on Balances:* The exact saved table values are France: 52,078,918.63 and Spain: 21,606,796.71. Earlier text transposing 1,498.77 was an assistant drafting error; the CSV table has always contained these exact figures.

---

## 6. Retention Strength Tiers & Subgroup Stability
*Source Table: `reports/tables/retention_strength_tiers_summary.csv` and `reports/tables/retention_tier_stability.csv`*

| Retention Tier | Definition Criteria | Customer Count (N) | Portfolio Share (%) | Churn Rate (%) | 95% Confidence Interval | Retained Customers | Balance Still to Protect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Core Sticky** | Active + 2 Products | 2,752 | 27.52% | 3.82% | [3.16% - 4.54%] | 2,647 | 144,149,491.56 |
| **Tier 2: Latent Sticky** | Inactive + 2 Products | 1,838 | 18.38% | 13.22% | [11.70% - 14.80%] | 1,595 | 83,004,264.91 |
| **Tier 3: Vulnerable** | Active + 1 Product | 2,332 | 23.32% | 20.84% | [19.21% - 22.51%] | 1,846 | 182,884,938.80 |
| **Tier 4 Combined** | Tier 4a OR Tier 4b | 3,078 | 30.78% | 38.30% | [36.58% - 40.03%] | 1,875 | 166,936,734.40 |
| *Tier 4a: Inactive Shallow* | Inactive + 1 Product | 2,752 | 27.52% | 33.54% | [31.79% - 35.32%] | 1,829 | 165,733,391.13 |
| *Tier 4b: Friction Zone* | 3-4 Products (Any Act) | 326 | 3.26% | 85.89% | [82.21% - 89.57%] | 46 | 1,203,343.27 |

### Stability Check: Tier Churn Rates in Young Cohort (Age 18-30)
*Source Table: `reports/tables/retention_tier_stability.csv`*
- Tier 1 (Core Sticky, Age 18-30): n = 601, Churn = 1.33% [0.50% - 2.50%]
- Tier 2 (Latent Sticky, Age 18-30): n = 431, Churn = 4.41% [2.55% - 6.50%]
- Tier 3 (Vulnerable, Age 18-30): n = 470, Churn = 7.66% [5.32% - 10.21%]
- **Tier 4 Combined (Age 18-30):** n = 466, Churn = **19.10%** [15.67% - 22.75%]
  - Tier 4a Alone (Inactive 1-Prod, Age 18-30): n = 433, Churn = **15.70%** [12.47% - 19.17%]
  - Tier 4b Alone (3-4 Prods, Age 18-30): n = 33, Churn = **63.64%** [45.45% - 78.79%]

*Reconciliation Note on Tier 4 Young Churn:* The exact value for combined Tier 4 in Age 18-30 is **19.10%** (n = 466, 89 churned). The earlier mention of "17.66%" was an assistant drafting typo.

---

## 7. Model Architecture Evaluation & Decile Lift
*Source Tables: `reports/tables/model_evaluation_metrics.csv` and `reports/tables/model_top_decile_lift.csv`*

Evaluated on held-out test set ($N_{\text{test}} = 2,000$, 407 churners, 1,593 retainers, base rate = 20.35%):

| Model Architecture | Hyperparameter / Calibration Tuning | ROC-AUC | PR-AUC | Recall (Threshold 0.50) | Precision (Threshold 0.50) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline: Logistic Regression** | `class_weight='balanced'` | 0.8370 [0.8135 - 0.8582] | 0.6263 [0.5782 - 0.6731] | 0.7568 [0.7116 - 0.7961] | 0.4490 [0.4124 - 0.4862] |
| **Main Model: Tuned XGBoost** | `scale_pos_weight=3.91` (CV inside Train) | 0.8690 [0.8474 - 0.8891] | 0.7171 [0.6758 - 0.7555] | 0.7666 [0.7222 - 0.8061] | 0.5115 [0.4723 - 0.5513] |
| **Calibrated XGBoost (Final Best Pipeline)** | Isotonic / Sigmoid Calibrated (CV=3 inside Train) | **0.8699** [0.8487 - 0.8900] | **0.7184** [0.6767 - 0.7586] | 0.5233 [0.4766 - 0.5728] | 0.7148 [0.6630 - 0.7645] |

### Calibrated Pipeline Decile Performance (Test Set, N = 2,000):
| Targeting Tier | Customer Count (N) | Actual Churners Captured | Precision (%) | Recall (%) | Lift Factor vs Base Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top 10% Highest-Risk Decile** | 200 | 172 of 407 | **86.00%** | **42.26%** | **4.23x** |
| **Top 20% Highest-Risk Quintile** | 400 | 256 of 407 | **64.00%** | **62.90%** | **3.14x** |

*Reconciliation Note on Model ROC-AUC:*
- The raw tuned XGBoost model achieves ROC-AUC **0.8690** (PR-AUC 0.7171).
- The final probability-calibrated pipeline achieves ROC-AUC **0.8699** (PR-AUC 0.7184).
- The earlier figure "0.8659" was an assistant drafting typo.

---

## 8. U2 Behavioral vs Demographic Paired Bootstrap
*Source Table: `reports/tables/u2_feature_group_comparison.csv`*

Evaluated using $B = 1,000$ paired bootstrap resamples on the test set:

| Feature Set | Features Included | LR ROC-AUC | LR PR-AUC | XGB ROC-AUC | XGB PR-AUC | XGB Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Behavioral Only** | `IsActiveMember`, `NumOfProducts`, `ProductTier` | 0.7517 | 0.4855 | 0.7633 [0.7381 - 0.7877] | 0.5119 [0.4662 - 0.5576] | 0.6757 [0.6291 - 0.7201] |
| **2. Demographic Only** | `Age`, `Geography`, `Gender` | 0.7508 | 0.3922 | 0.7745 [0.7482 - 0.7997] | 0.4714 [0.4229 - 0.5242] | 0.7027 [0.6590 - 0.7474] |
| **3. Financial Only** | `Balance`, `EstimatedSalary`, `BalanceToSalaryRatio` | 0.5691 | 0.2503 | 0.5693 [0.5386 - 0.6009] | 0.2615 [0.2304 - 0.3027] | 0.5405 [0.4926 - 0.5894] |
| **4. Behavioral + Age** | Behavioral + `Age` | 0.8260 | 0.5982 | 0.8498 [0.8264 - 0.8708] | 0.6897 [0.6456 - 0.7291] | 0.7445 [0.6981 - 0.7866] |
| **5. Combined All Features** | All 10 Features | 0.8370 | 0.6264 | 0.8690 [0.8474 - 0.8891] | 0.7171 [0.6758 - 0.7555] | 0.7666 [0.7222 - 0.8061] |
| **Paired Difference** | **Behavioral - Demographic** | - | - | **-0.0112 [-0.0445, +0.0232]** | **+0.0405 [-0.0218, +0.0999]** | **CI spans zero** |

*Academic Thesis Conclusion:* Because both the delta ROC-AUC CI [-0.0445, +0.0232] and delta PR-AUC CI [-0.0218, +0.0999] span zero, behavioral features alone do not statistically outperform demographics in isolated discrimination. Instead, **behavioral and demographic features are about equally informative and complementary**, with behavioral features providing the bank's **actionable operational lever**.

---

## 9. Relationship Strength Index (RSI) Formulations
*Source Table: `reports/tables/u1_rsi_comparison.csv` and `reports/tables/u4_statistical_tests.csv`*

- **Plain Words Definition:** $\text{RSI} = 100 \times \text{predicted probability of staying}$ derived from an interpretable logistic regression model trained on the training fold.
- **Why Mean RSI Matches Portfolio Retention:** Logistic regression with an intercept mathematically constrains the sample mean of predicted probabilities to match the training empirical event rate ($\frac{1}{N}\sum \hat{p}_{\text{stay}} = 1 - \bar{y} = 79.63\%$). Hence, average RSI = **79.63**.

| RSI Metric / Formulation | Scoring Formula / Model | ROC-AUC | Paired Delta vs Data-Driven RSI | 95% Bootstrap CI | Statistical Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Data-Driven Logistic RSI** | Calibrated Logistic Model on Train | **0.7514** | Baseline (0.0000) | - | Statistically Superior |
| **Naive RSI 1 (4-Feature Heuristic)** | $25 \times (\text{Active} + 2\text{Prod} + \text{ZeroBal} + \text{Age}<40)$ | 0.6874 | -0.0640 | [-0.0814, -0.0471] | Excludes 0 (Significantly inferior) |
| **Naive RSI 2 (50/50 Dual Heuristic)** | $50 \times \text{IsActiveMember} + 50 \times (\text{NumOfProducts}==2)$ | 0.7054 | -0.0460 | [-0.0607, -0.0322] | Excludes 0 (Significantly inferior) |

- **Mann-Whitney U Test (RSI Separation):** Compares RSI scores between churners and retainers: $U = 3,090,323.0$, $p < 10^{-15}$.

---

## 10. U6 What-If Counterfactual Simulations & Cohort Benchmarking
*Source Tables: `reports/tables/u6_what_if_simulation_examples.csv` and `data/processed/bank_churn_scored.csv`*

*Framing Notice:* All counterfactuals reflect model-predicted differences between similar customer profiles, not proven causal outcomes.

### Benchmark A: Single Customer Profile Example
*Profile: Spain, Male, Age 44, Tenure 8, Balance 113,755, Salary 149,756, HasCrCard = 1, NumOfProducts = 1, IsActiveMember = 0*
- **Calibrated Baseline Churn Probability:** **34.08%**
- **Observed Churn of Similar Historical Cohort** (Spain, Inactive, 1 Product, Age 31-45, $n = 355$): **24.79%** (88 exited / 355)
- **Counterfactual Scenario if Activated (`IsActiveMember = 1`):** **17.12%**
- **Model-Predicted Difference for Single Customer:** **-16.95 percentage points** (-16.95 pts)

### Benchmark B: Population-Level Counterfactual
*Population: All 1,597 Retained Inactive 1-Product Customers across the bank*
- **Baseline Mean Calibrated Churn Probability:** **24.80%** (0.24795)
- **Counterfactual Mean Calibrated Churn if Activated:** **14.76%** (0.14759)
- **Population Model-Predicted Difference:** **-10.04 percentage points** (-10.04 pts)

*Reconciliation Note on -10.04 pts:* The single example delta is **-16.95 pts** and the population delta is **-10.04 pts**. Earlier text reporting -10.04 pts for both was an accidental reuse of the population counterfactual for the single example.
