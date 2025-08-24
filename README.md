# Stock Price Forecasting with ARIMA and SARIMA

## 📌 Project Overview
This project applies **ARIMA** and **SARIMA** models to forecast stock price time series data.

The workflow includes:
- Downloading historical stock data using `yfinance`
- Performing **Stationarity checks** with the Augmented Dickey-Fuller (ADF) test
- Using `pmdarima.auto_arima` to automatically select the best ARIMA/SARIMA parameters
- Training models with `statsmodels.SARIMAX`
- Evaluating forecast performance with error metrics (RMSE and MAE)
- Comparing forecasts from ARIMA and SARAIMA against actual test data


## ⚙️ Tech Stack
- **Python 3.12**
- **Libraries**:
  - `numpy`
  - `matplotlib`
  - `yfinance`
  - `statsmodels`
  - `pmdarima`
  - `scikit-learn`
 

## 📊 Methodology
1. **Data Collection**
   - Downloaded daily stock price data (Mon-Fri, skipping holidays)
   - Filled missing dates with forward-fill to maintain business day frequency
  
2. **Staionarity Check**
   - Applied Augmented Dicky-Fuller (ADF) test to detect unit root
   - If p-value > 0.05 -> data is non-stationary, differencing applied

3. **Model Selection**
   - **ARIMA**: `auto_arima(seasonal=False)`
   - **SARIMA**: `auto_arima(seasonal=True, m=5)` (assuming weekly seasonality with 5 trading days)

  4. **Train-Test Split**
     - Training data: all but last 60 observations
     - Test data: last 60 observations
    
  5. **Model Fitting and Forecasting**
     - Fitted ARIMA and SARIMA models on training data
     - Foarecasted values for the test horizon

  6. **Evaluation**
     - Computed error metrics:
       - **Root Mean Square Error (RMSE)**
       - **Mean Absolute Error (MAE)**
      - Results:
        ```
        ARIMA -> RMSE: 7.29, MAE: 6.26
        SARIMA -> RMSE: 7.17, MAE: 6.15
        ```

## 📈 Results
- **ARIMA (0,1,1)** produced a **flat forecast**, behaving like a differenced random walk
- **SARIMA (3,1,2)(2,0,0,5)** captured some seasonal structure but still flattened over longer horizons
- Metrics (RMSE and MAE) suggest SARIMA slightly outperformed ARIMA on this dataset

<img width="831" height="451" alt="output" src="https://github.com/user-attachments/assets/81f77389-8b66-413b-8c0e-d576e726e90b" />


## 📚 Next Steps
- Add a drift term to ARIMA to allow non-flat forecasts
- Ty modelling log-returns instead of raw prices
- Experiement with alternative models (Holt-Winters, Prophet, LSTM)

