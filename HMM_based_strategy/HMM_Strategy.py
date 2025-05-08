'''
HMM strategy
'''
# Dont forget to install hmmlearn

import numpy as np
import pandas as pd
import yfinance as yf
import hmmlearn.hmm as hmm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
from warnings import filterwarnings
filterwarnings('ignore')

try:
  df = yf.Ticker("MSFT").history(period="1d", start="2020-01-01", end="2026-01-01")

  if df.empty:
        raise ValueError("No data retrieved from Yahoo Finance")
except Exception as e:
    print(f"Data retrieval failed: {e}")
    exit()

df.drop(columns=["Stock Splits", "Dividends"], inplace=True)

total_size = len(df)
train_size = int(total_size * 0.7)
gap_size = max(int(total_size * 0.1), 24)
test_size = total_size - train_size - gap_size

if test_size <= 30:
    raise ValueError("Insufficient data for meaningful backtesting")

# data split
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size + gap_size:].copy()

# Creating features
def create_features(data, train=True):
    data = data.copy()
    lookback = 6

    if len(data) < lookback + 1:
        raise ValueError(f"Need at least {lookback+1} periods for features")

    data["Return"] = data["Close"].pct_change().shift(1)
    data["Volatility"] = data["Return"].rolling(5, min_periods=1).std()
    data["Momentum"] = data["Close"].shift(1) - data["Close"].shift(6)
    data["Log_Volume"] = np.log(data["Volume"].shift(1) + 1e-6)

    return data.dropna() if train else data

try:
    train_df = create_features(train_df, train=True)
    test_df = create_features(test_df, train=False).dropna()
except ValueError as e:
    print(f"Feature creation failed: {e}")
    exit()

features = ["Return", "Volatility", "Momentum", "Log_Volume"]
scaler = StandardScaler()
# Scaling
try:
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
except ValueError as e:
    print(f"Scaling failed: {e}")
    exit()
#model and training
model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=10,
    tol=1e-6,
    verbose=False
)

try:
    model.fit(X_train)
    with open("hmm_model_MSFT.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler_MSFT.pkl", "wb") as f:
        pickle.dump(scaler, f)
except Exception as e:
    print(f"Model training failed: {e}")
    exit()



def rolling_predict(model, train_data, test_data, window_size=30):
    predictions = []

    initial_window = train_data[-window_size:] if len(train_data) >= window_size else train_data
    current_window = initial_window.copy()

    for test_point in test_data:
        current_window = np.vstack([current_window, test_point])
        if len(current_window) > window_size:
            current_window = current_window[-window_size:]

        try:
            state = model.predict(current_window)[-1]
        except Exception as e:
            print(f"Prediction failed: {e}")
            state = 0
        predictions.append(state)

    return predictions

test_states = rolling_predict(model, X_train, X_test)
test_df["Hidden_State"] = test_states
initial_cash = 10000
cash = initial_cash
position = 0
trade_log = []
last_valid_price = None

for i in range(1, len(test_df)):
    current_row = test_df.iloc[i]
    prev_row = test_df.iloc[i-1]


    if np.isnan(current_row["Close"]) or current_row["Close"] <= 0:
        continue

    prev_state = prev_row["Hidden_State"]
    curr_state = current_row["Hidden_State"]
    close_price = current_row["Close"]
    last_valid_price = close_price


    if prev_state == 0 and curr_state == 1 and cash > 0:
        position = cash / close_price
        cash = 0
        trade_log.append(("BUY", current_row.name, close_price))
    elif prev_state == 1 and curr_state == 2 and position > 0:
        cash = position * close_price
        position = 0
        trade_log.append(("SELL", current_row.name, close_price))


if position > 0 and last_valid_price is not None:
    cash = position * last_valid_price
    position = 0

final_value = cash + position * (last_valid_price if last_valid_price else 0)
roi = (final_value - initial_cash) / initial_cash * 100


plt.figure(figsize=(14, 7))
for state in sorted(test_df["Hidden_State"].unique()):
    subset = test_df[test_df["Hidden_State"] == state]
    if not subset.empty:
        plt.scatter(subset.index, subset["Close"], label=f"State {state}", s=10)


for i in range(1, len(trade_log)):
    try:
        prev_trade = trade_log[i-1]
        curr_trade = trade_log[i]

        if prev_trade[0] == "BUY" and curr_trade[0] == "SELL":
            plt.plot([prev_trade[1], curr_trade[1]],
                     [prev_trade[2], curr_trade[2]],
                     color="green" if curr_trade[2] > prev_trade[2] else "red",
                     linestyle="--", alpha=0.7)
    except Exception as e:
        print(f"Error plotting trade {i}: {e}")


plt.legend()
plt.title("ETH Hidden States with Trades (Strictly Time-Aware Backtest)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.show()


print("\nTrade Execution Summary:")
print(f"{'Action':<6} {'Date':<25} {'Price (USD)':<12}")
for trade in trade_log:
    print(f"{trade[0]:<6} {trade[1].strftime('%Y-%m-%d %H:%M'):<25} ${trade[2]:<10.2f}")

print(f"\nFinal Portfolio Value: ${final_value:.2f}")
print(f"ROI: {roi:.2f}%")