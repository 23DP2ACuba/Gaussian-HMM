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

2. **Run the MSFT Trainer + Backtester**

```bash
python hmm_strategy.py
```

This will:
- Download `MSFT` data
- Train an HMM model
- Save the model and scaler
- Simulate trading and print ROI and trade details
- Display trade visualization

3. **Run the ETH Tester (using pre-trained model)**

Ensure `hmm_model_ETH.pkl` and `scaler_ETH.pkl` are placed in the expected path (`/content/`) or update the path in `hmm_tester.py`.

```bash
python hmm_tester.py
```

---

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

## 📌 Notes

- Historical data range: `2020-01-01` to `2026-01-01`
- Trained model is time-aware and avoids data leakage using a gap between train/test
- Model retraining can be done on any asset by updating ticker symbol and filenames

---

## 🛡️ Disclaimer

This code is for educational and research purposes only. It is **not financial advice**. Historical performance does not guarantee future results.

---

## 📬 Contact

For questions, reach out or fork this repo to contribute improvements.
