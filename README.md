# Gaussian-HMM
---

## HMM-Based Trading Strategy

This project implements a **Hidden Markov Model (HMM)** strategy for time-series financial data using stock/crypto price history. It involves training a Gaussian HMM to learn hidden market states from Microsoft stock (`MSFT`) and applying the model to backtest trading performance on both MSFT and Ethereum (`ETH-USD`) price data.

---

## 📊 Features

- Downloads historical price and volume data from Yahoo Finance using `yfinance`
- Extracts technical features: return, volatility, momentum, and log volume
- Trains a Gaussian HMM on historical data
- Implements time-aware rolling prediction
- Executes simple buy/sell strategy based on predicted state transitions
- Visualizes price movements, state segmentation, and trade performance
- Calculates Return on Investment (ROI) and trade statistics

---

## 📁 Project Structure

```bash
.
├── hmm_strategy.py        # HMM training + backtest on MSFT
├── hmm_tester.py          # Loads pretrained model and tests on ETH-USD
├── hmm_model_MSFT.pkl     # Trained HMM model for MSFT (output)
├── scaler_MSFT.pkl        # Scaler used for MSFT feature normalization (output)
├── hmm_model_ETH.pkl      # Trained HMM model for ETH (expected for testing)
├── scaler_ETH.pkl         # Scaler for ETH data (expected for testing)
└── README.md              # This file
```

---

## 🔧 Setup Instructions

1. **Install Dependencies**

```bash
pip install numpy pandas yfinance hmmlearn scikit-learn matplotlib
```

2. **Run the HMM Trainer + Backtester**

```bash
python hmm_strategy.py
```

## 📈 Trading Logic

The strategy relies on transitions between hidden states:

- **Buy Signal:** Transition from State 0 ➝ State 1
- **Sell Signal:** Transition from State 1 ➝ State 2

The strategy assumes a simple model with 3 market states learned from the data. It simulates full-position trades (all-in/all-out) without leverage or fees.

---

## 📉 Output Metrics

Both scripts provide:

- Trade log with timestamps and prices
- Final portfolio value
- ROI (%)
- Number of profitable and losing trades (in `hmm_tester.py`)

---
