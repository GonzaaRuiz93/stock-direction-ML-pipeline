
<center>

# Stock Direction Prediction Pipeline (LR + RF)

</center>

## Overview

This project implements an end-to-end machine learning pipeline to predict daily stock direction using a two-stage classification system:

- Stage 1: Logistic Regression (broad signal filter)
- Stage 2: Random Forest (signal refinement)

The system is evaluated using time-series cross-validation and trading-oriented metrics.

The objective is not to maximize raw accuracy, but to design a robust and realistic trading signal framework under weak predictability conditions.

<br>

## Modeling Philosophy

Financial time series are noisy and weakly predictable.

Instead of relying on a single complex model, this project uses:

1. Logistic Regression — Entry Filter

    - High recall
    - Captures most potential upward movements
    - Reduces risk of overfitting due to model simplicity

2. Random Forest — Signal Refinement
 
    - Applied only to filtered candidates
    - Reduces false positives
    - Improves signal robustness

This architecture mimics real-world quantitative filtering pipelines.

<br>

## Features Used

All features are computed using daily closing prices:

- ret_1: 1-day return
- ret_5: 5-day return
- ma_5: 5-day moving average
- vol_5: 5-day rolling volatility

Target variable:
                   
    1 -> if next-day return > 0
    0 -> otherwise

<br>

## Validation Methodology

To avoid data leakage and ensure realistic evaluation:

- TimeSeriesSplit (5 folds) is used
- No shuffling
- Forward-chaining validation
- Last row removed due to target shifting

<br>

## Evaluation Metrics

Metrics are computed per fold and averaged:

- Mean Precision
- Precision Std
- Mean Recall
- Mean Trades
- Mean ROC-AUC
- ROC-AUC Std

The focus is on:

- Stability
- Trade selectivity
- Balance between recall and precision

<br>

## Results Interpretation

Typical observed behavior:

| Metric | Logistic Regression | LR + RF |
| --------- | ------------------- | ----------------------- |
| Precision | Moderate (~0.55) | Slightly Higher (~0.56) |
| Precision Std | Moderate (~0.03) | Slightly Lower (~0.02) |
| Recall | High (~0.90) | Lower (~0.50) |
| AUC | Weak (~0.53) | Slightly Lower (~0.52) |
| AUC Std | High (~0.05) | Lower (~0.02) |

Interpretation:

- Logistic Regression captures most opportunities.
- Random Forest reduces trade frequency.
- Combined model improves stability.
- No signs of overfitting. 
- Performance levels are consistent with the weak signal structure typically observed in daily equity returns.


<br>

## Project Structure
           
    main.py
    data.py
    models.py
    gen_reporte.py
    visualization.py

Pipeline flow:

    Data
	Feature Engineering
	Target Creation
    TimeSeries CV
	LR Filter
	RF Refinement
    Metrics
	Visualization
	Report

<br>

## How to Run

    python main.py

Parameters can be adjusted inside main.py:

    TICKER = "AAPL"
    UMBRAL_LR = 0.45
    UMBRAL_RF = 0.55

<br>

## Future Work & Scalability

- Implement full backtesting engine
- Add transaction cost modeling
- Add SHAP feature importance
- Hyperparameter optimization
- Add multi-asset support
- Deploy as a daily automated pipeline