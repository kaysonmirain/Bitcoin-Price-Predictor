# Bitcoin Price Predictor

A real-time Bitcoin price tracking and short-term prediction dashboard using live market data.

## Features
* **Live Data Feed:** Fetches BTC/USDT data directly from Binance via the CCXT library.
* **Dynamic Visualization:** Real-time animated charts showing price paths and "Next Target" markers.
* **Predictive Analysis:** Uses linear regression on a 5-minute interval to forecast upcoming price movements.
* **Performance Metrics:** Real-time tracking of prediction accuracy and Average Mean Error (MAE).

## Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install ccxt pandas numpy matplotlib
