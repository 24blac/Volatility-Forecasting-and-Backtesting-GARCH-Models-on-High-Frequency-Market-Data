import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas_ta as ta
import plotly.graph_objects as go

# Load the data from a CSV file and parse the 'time' column as a datetime object
data = pd.read_csv("../../../Documents/DerivData/Boom 1000 Index_m1.csv", index_col=0)
data["time"] = pd.to_datetime(data["time"], unit="s")

# Filter data for the months between June and August (6 to 8)
data["month"] = data["time"].dt.month
data = data[(data["month"] > 5) & (data["month"] < 9)].copy()

# Drop irrelevant columns: tick_volume, spread, real_volume
data.drop(columns = ["tick_volume", "spread", "real_volume"], inplace=True)

# Further filter data for July and restrict to the first 6 days
data = data[data["month"]==7]
data = data[data["time"].dt.day < 7].reset_index(drop=True)

# Display the first few rows of the filtered data
data.head()

# Calculate returns as percentage change in closing prices
data["returns"] = 100 * data["close"].pct_change().dropna()
returns = 100 * data["close"].pct_change().dropna()

# Calculate log returns
log_returns = np.log(data["close"]).diff().dropna()

# Plot histogram of log returns
plt.figure(figsize=(10,4))
log_returns.hist(bins=30)
plt.ylim(0, 125)

# Plot log returns over time
plt.figure(figsize=(10,4))
plt.plot(log_returns)
plt.ylabel("Pct Change", fontsize=16)
plt.title("Boom 1000 Index Returns", fontsize=20)

# Plot ACF and PACF of squared log returns to evaluate volatility clustering
plot_acf(log_returns**2)
plt.ylim(-0.25, 0.25)
plt.show()

plot_pacf(log_returns**2)
plt.ylim(-0.25, 0.25)

# Scale returns for fitting the GARCH model
scaled_returns = returns * 100

# Define a GARCH model with specified p and q parameters and fit it to the data
model = arch_model(scaled_returns[:121], p=9, q=7)
model_fit = model.fit(disp="off")

# Display the model summary
model_fit.summary()

# Function to fit GARCH models for given p and q values
def fit_garch(scaled_returns, p, q):
    model = arch_model(scaled_returns, vol='GARCH', p=p, q=q)
    try:
        results = model.fit(disp='off')
        return results
    except:
        return None

# Function to check if p-values of fitted model parameters are statistically significant
def check_pvalues(results, significance_level=0.05):
    if results is None:
        return False
    params = results.params.iloc[1:]  # Exclude constant
    pvalues = results.pvalues.iloc[1:]  # Exclude constant
    return any(pvalue < significance_level for pvalue in pvalues)

# Find the best GARCH model by iterating over different p and q combinations
def find_best_model(scaled_returns, max_p=11, max_q=11, significance_level=0.05):
    best_model = None
    best_aic = np.inf
    best_bic = np.inf
    best_pq = None
    all_results = {}

    # Iterate over combinations of p and q values
    for p, q in product(range(1, max_p + 1), range(0, max_q + 1)):
        results = fit_garch(scaled_returns, p, q)
        all_results[(p, q)] = results

        # If results are significant, track the best model based on AIC
        if results is not None and check_pvalues(results, significance_level):
            if results.aic < best_aic:
                best_model = results
                best_aic = results.aic
                best_bic = results.bic
                best_pq = (p, q)

    return best_model, best_pq, best_aic, best_bic, all_results

# Find and print the best GARCH model based on AIC and BIC criteria
best_model, best_pq, best_aic, best_bic, all_results = find_best_model(scaled_returns)

if best_model is not None:
    print(f"Best model: GARCH{best_pq}")
    print(f"AIC: {best_aic:.4f}")
    print(f"BIC: {best_bic:.4f}")
    print("\nModel Summary:")
    print(best_model.summary().tables[1])
else:
    print("No suitable model found. Here are the details for all fitted models:")
    for (p, q), results in all_results.items():
        if results is not None:
            print(f"\nGARCH({p},{q}):")
            print(f"AIC: {results.aic:.4f}")
            print(f"BIC: {results.bic:.4f}")
            print("P-values:")
            print(results.pvalues)
        else:
            print(f"\nGARCH({p},{q}): Failed to converge")

# If a model was found, plot its conditional volatility
if best_model is not None:
    best_model.plot()
    plt.show()

# Generate rolling predictions of volatility over a test size window
rolling_predictions = []
test_size = 300

# Rolling window forecast loop
for i in range(test_size):
    train = scaled_returns[:-(test_size-i)]
    model = arch_model(train, p=9, q=7)
    model_fit = model.fit(disp="off")
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

# Convert rolling predictions to pandas series and plot
rolling_predictions = pd.Series((rolling_predictions), index=scaled_returns.index[-test_size:])
plt.figure(figsize=(10,4))
true, = plt.plot(scaled_returns[-test_size:])
preds, = plt.plot(rolling_predictions)
plt.title("Volatility - Rolling Forecast", fontsize=20)
plt.legend(["True Returns", "Predicted Volatility"], fontsize=10)

# Visualize percentage changes in rolling predictions
plt.figure(figsize=(10,4))
rolling_predictions.pct_change().plot()

# Identify spikes in risk (rolling prediction changes above threshold)
spike_threshold = 0.25
spike_condition = rolling_predictions.pct_change() > spike_threshold
len(spike_condition[spike_condition==True])
spike_condition[spike_condition==True]

# Prepare data for trade logic by merging predictions with original sample
sample = data[-test_size:]
predictions = pd.Series(rolling_predictions, name="predictions")
merge = sample.merge(predictions, on=sample.index, how="left")
merge["pred_chng"] = merge["predictions"].pct_change()
merge.info()

# Initialize an empty trades DataFrame to store trading decisions
trades = pd.DataFrame(columns=['state', 'order_type', 'open_time', 'open_price', 'close_time', 'close_price'])

# Trading logic based on volatility spike conditions
for i, x in merge.iterrows():
    # Open trade logic: open a buy trade if conditions are met
    num_open_trades = trades[trades['state'] == 'open'].shape[0]
    if (x["pred_chng"] >= 0.25) and (num_open_trades==0):
        trades.loc[len(trades), trades.columns] = ['open', 'buy', x['time'], x['open'], None, None]

    # Close trade logic: close the trade if price hits high or low conditions
    open_trades = trades[trades['state'] == 'open'].shape[0]
    if open_trades == 0:
        continue

    stop_loss = trades["open_price"].iloc[-1] - 2.5
    close_cond1 = x["high"] > x["open"]
    close_cond2 = x['low'] <= stop_loss

    if close_cond1:
        trades.loc[trades['state'] == 'open', ['state', 'close_time', 'close_price']] = ['closed', x['time'], x['high']]
    if close_cond2:
        trades.loc[trades['state'] == 'open', ['state', 'close_time', 'close_price']] = ['closed', x['time'], x['low']]

# Calculate trade returns and cumulative returns
trades["returns"] = trades["close_price"] - trades["open_price"]
trades["cum_returns"] = trades["returns"].cumsum()
trades = trades[:-1]  # Remove the last incomplete trade

# Plot cumulative returns over time
(trades["cum_returns"]*0.2).plot()

# Plot candlestick chart of the data
candles = go.Candlestick(x=merge["time"],
                        open=merge["open"],
                        high=merge["high"],
                        low=merge["low"],
                        close=merge["close"])
fig = go.Figure(data=candles)
fig.update_layout(xaxis_rangeslider_visible=False, height=700)

# Visualize the trades on the candlestick chart
for i, trade in trades.iterrows():
    color = 'yellow' if trade['returns'] >
