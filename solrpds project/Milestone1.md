2\. Project Evolution and Problem Setting

The project was initially designed to focus on temporal graph mining and coordination detection, including community detection and pattern analysis. However, during implementation, several practical challenges required us to refine our approach.

First, there is a clear schema mismatch between datasets. The SolRPDS dataset (2021–2024) provides aggregated liquidity pool features, whereas the 2025 dataset collected from Dune consists of raw transaction-level swap data. As a result, many original features cannot be directly reused. Second, there is label inconsistency. Historical data contains labels derived from liquidity behavior, while the 2025 dataset only allows heuristic labeling based on observable activity patterns. Therefore, the evaluation in 2025 cannot be interpreted as true predictive performance. Third, there is a severe distribution shift. The proportion of rug pull tokens in historical data is significantly higher than in the 2025 dataset, leading to strong imbalance and model miscalibration.

To address these issues, we adopt a hybrid approach that combines feature-based modeling, graph-based representation, and cross-time analysis.

3\. Methodology

3.1 Shared Feature Construction

We construct a shared feature space across both datasets using general behavioral signals:

Lifespan: duration between first and last activity

Activity count: number of interactions or trades

Total volume: proxy for liquidity or trading intensity

Imbalance ratio: asymmetry between buy and sell behavior

Concentration ratio: top wallet share (2025) or approximated by top liquidity-pool share (historical)

These features provide a consistent representation while accommodating differences in data structure.



3.2 Graph-Based Representation

We construct bipartite graphs to capture relationships between tokens and interacting entities:

Historical data: token ↔ liquidity pool

2025 data: token ↔ wallet

From these graphs, we extract structural features such as:

node degree, representing the number of connected wallets or pools

number of connected entities

Basic graph features are integrated into the model. However, community-level features and full graph neural network (GNN) training are not yet incorporated into the cross-time prediction pipeline. Graph mining is currently applied at the feature level, while more advanced graph-based learning methods are deferred to future work. 



3.3 Cross-Time Machine Learning

We evaluate generalization using a temporal split:

Train: 2021–2023

Validate: 2024

Apply (analysis only): 2025

We train three models: Logistic Regression, Random Forest, and XGBoost

On the 2024 validation set, the models achieve high recall and F1 scores in the range of 0.75–0.79, while precision and accuracy remain moderate. This indicates that although the models capture many rug pull cases, they also produce a significant number of false positives. While the current experiments are conducted on data up to April 2025, the pipeline is designed to support incremental data ingestion and can be extended to cover the full 2025 period. This allows for longitudinal analysis of distribution shift and provides a foundation for continuous monitoring of evolving fraud patterns in the Solana ecosystem. 



3.4 Threshold Sensitivity Analysis

To better understand model behavior under distribution shift, we evaluate multiple prediction strategies:

default probability threshold (0.5)

threshold optimized for F1 in 2024

top-k selection (top 1%, 5%, 10% highest risk)

This allows us to distinguish between classification performance and ranking behavior.



3.5 Distribution Shift Analysis

We quantify feature drift using:

Kolmogorov - Smirnov (KS) statistic

standardized mean difference

Features are categorized into low, medium, and high drift, helping identify which features contribute most to performance degradation.



3.6 Data Pipeline (Conceptual Layers)

Bronze Layer:  Raw Data Ingestion

Load SolRPDS historical dataset and the Dune transaction data

Perform cleaning, deduplication, and schema normalization

Silver Layer: Feature Construction

Aggregate transaction-level data into token-level features

Compute shared behavioral features

Apply heuristic labeling for 2025

Split historical data temporally

Gold Layer: Modeling and Analysis

Train models on historical data

Validate in 2024

Apply models to 2025 for risk scoring

Perform threshold sensitivity and drift analysis



4\. Data and Experimental Evidence

4.1 Dataset Overview

To ensure reproducibility and transparency, we summarize the key statistics of both datasets used in this study.

Historical dataset (SolRPDS, 2021–2024):

Metric

Value

Total tokens

35,320

Rug tokens

19,464

Non-rug tokens

15,856

Rug rate

55.1%







2025 dataset (Dune, Jan–Apr 2025):

Metric

Value

Total tokens

37,015

Heuristic rug tokens

903

Heuristic non-rug tokens

36,112

Heuristic rug rate

2.4%

Raw transactions

80,000









The 2025 dataset is aggregated from four monthly partitions (Jan–Apr), highlighting a major distribution shift: rug rate drops from 55.1% to 2.4%.



4.2 Feature Construction and Definitions

To enable cross-time modeling, we define a shared feature space that can be consistently computed across both datasets. Key features include:

Imbalance ratio

Historical: TOTAL\_REMOVED\_LIQUIDITY / TOTAL\_ADDED\_LIQUIDITY

2025: sell\_volume\_usd / buy\_volume\_usd

Concentration ratio

Historical: top liquidity pool volume share

2025: top wallet volume share

Activity count

Historical: number of liquidity add/remove events

2025: number of token-side swap events

These definitions ensure that features capture analogous behavioral signals, even though their underlying data sources differ.



4.3 Graph Structure

To support graph-based analysis, we construct token–entity interaction graphs from the 2025 dataset.

Graph Component

Count

Token nodes (aligned)

37,015

Wallet/entity nodes

56,178

Raw edge rows

160,000

Unique token-wallet edges

69,200

Filtered token interactions

75,716





These statistics confirm that the graph is non-trivial in scale, with tens of thousands of nodes and edges. In this study, graph information is incorporated through structural features such as node degree (number of connected wallets or pools). More advanced graph analysis methods, such as community detection, are prepared but not yet fully integrated into the cross-time prediction pipeline. Graph-derived features such as node degree and number of connected wallets are included in the model feature set, providing a structural view of token interaction patterns. 



4.4 Model Performance on Historical Data (2024 Validation)

We evaluate model performance using a temporal split, where models are trained on 2021–2023 and validated on 2024.

Model

Accuracy

Precision

Recall

F1

ROC-AUC

Logistic Regression

0.718

0.696

0.918

0.792

0.695

Random Forest

0.656

0.649

0.896

0.753

0.724

XGBoost

0.676

0.648

0.974

0.779

0.746



Logistic Regression achieves the highest F1 score (0.792), although the difference across models is relatively small. It is selected as the primary model due to its interpretability and stable performance. Overall, the models achieve high recall but only moderate precision and accuracy, indicating that while many rug tokens are correctly identified, a substantial number of false positives remain. Preliminary inspection of feature importance indicates that lifespan, concentration ratio, and imbalance ratio are among the most influential predictors across models. 



Model

Top Features

Logistic Regression

activity\_count, graph\_degree, connected\_entities, imbalance\_ratio, lifespan\_hours

Random Forest

imbalance\_ratio, total\_volume, lifespan\_hours, activity\_count, entity\_concentration\_ratio

XGBoost

imbalance\_ratio, activity\_count, total\_volume, connected\_entities, graph\_degree





4.5 Cross-Time Prediction Behavior on 2025 dataset

We apply the trained models to the 2025 dataset using the threshold optimized on 2024 validation.

Model

Predicted Rug Rate (2025)

Logistic Regression

74.6%

Random Forest

99.9%

XGBoost

99.9%



These predicted rates are significantly higher than the heuristic rug rate of 2.4%, indicating strong miscalibration under distribution shift. Since 2025 lacks ground-truth labels, results reflect prediction behavior rather than true performance. Therefore, these results should be interpreted as prediction behavior and heuristic agreement, rather than true out-of-sample performance.



4.6 Key Observations

The results show a severe class distribution shift between historical and 2025 data, causing models to over-predict the positive class. Despite strong validation performance, cross-time application results in poor transfer as classifiers. While graph features are included, their contribution remains at a basic structural level.



4.7 Result Interpretation

The results reveal a clear gap between in-distribution performance and cross-time behavior. While models achieve high recall and F1 on the 2024 validation set, their predictions on 2025 are heavily skewed toward the positive class, with up to 99.9% of tokens classified as high risk. This indicates that the models capture patterns specific to the historical distribution but fail to adapt under new conditions.

This miscalibration is driven by both feature drift and class distribution shift. In historical data, rug pulls account for over half of all samples, whereas in 2025, they represent only a small fraction. At the same time, key features such as volume, activity, and imbalance change significantly, causing learned decision boundaries to lose meaning.

Although some limited ranking signal remains under top-k selection, it is weak and inconsistent. Overall, these results show that models trained on historical blockchain data cannot be directly applied to future data without recalibration or adaptation.

5\. Discussion

The performance gap is driven by differences in feature semantics, changes in class distribution, label inconsistency, and evolving attacker behavior. Although classification performance degrades, models retain limited ranking capability, suggesting potential use in risk prioritization rather than direct prediction.



6\. Limitations and Future Work

This study has several limitations:

2025 labels are heuristic and not ground truth

Feature alignment simplifies the original datasets

Community-level graph features are not yet integrated

Future work includes integrating community-level graph features, exploring anomaly detection, and improving labeling strategies.



