import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
import threading
import time
from datetime import datetime, timedelta

exchange = ccxt.binance()
symbol_display = 'BTC/USD'
symbol_fetch = 'BTC/USDT'

df = pd.DataFrame()
predictions = {}
prediction_history_list = []
accuracy_history = []
mae_history = []
data_lock = threading.Lock()
data_ready = False

utc_offset = datetime.now() - datetime.utcnow()

def fetch_data():
    global df, data_ready, accuracy_history, mae_history, prediction_history_list
    ohlcv = exchange.fetch_ohlcv(symbol_fetch, '5m', limit=100)
    with data_lock:
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms') + utc_offset
    data_ready = True

    while True:
        try:
            new_ohlcv = exchange.fetch_ohlcv(symbol_fetch, '5m', limit=5)
            new_df = pd.DataFrame(new_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            new_df['ts'] = pd.to_datetime(new_df['ts'], unit='ms') + utc_offset
            
            with data_lock:
                df = pd.concat([df, new_df]).drop_duplicates('ts').tail(100)
                current_ts = df['ts'].iloc[-1]
                
                if current_ts in predictions:
                    actual = df['close'].iloc[-1]
                    predicted = predictions[current_ts]
                    diff = abs(actual - predicted)
                    volatility = df['close'].tail(20).std()
                    acc = max(0, 100 * np.exp(-diff / (volatility * 2))) if volatility > 0 else 100
                    accuracy_history.append(acc)
                    mae_history.append(diff)
                    if len(accuracy_history) > 60:
                        accuracy_history.pop(0)
                        mae_history.pop(0)

                y = df['close'].tail(5).values
                x = np.arange(len(y))
                slope, intercept = np.polyfit(x, y, 1)
                next_ts = current_ts + pd.Timedelta(minutes=5) 
                pred_val = (slope * len(y)) + intercept
                
                predictions[next_ts] = pred_val
                prediction_history_list.append((next_ts, pred_val))
                if len(prediction_history_list) > 100:
                    prediction_history_list.pop(0)
                
            time.sleep(30)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

threading.Thread(target=fetch_data, daemon=True).start()

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 7))
fig.canvas.manager.set_window_title('Bitcoin Price Predictor')

def animate(i):
    if not data_ready or df.empty: return
    
    with data_lock:
        local_df = df.copy()
        local_hist = list(prediction_history_list)
        avg_acc = np.mean(accuracy_history) if accuracy_history else 0
        avg_mae = np.mean(mae_history) if mae_history else 0
    
    ax.clear()
    now = datetime.now()
    last_ts = local_df['ts'].iloc[-1]
    last_price = local_df['close'].iloc[-1]
    
    ax.plot(local_df['ts'], local_df['close'], color='#1f77b4', linewidth=2, label='Actual Path')
    ax.axvline(x=last_ts, color='#ffffff', linestyle='--', alpha=0.5, linewidth=1, label='Current Entry')

    past_pts = [pt for pt in local_hist if pt[0] <= last_ts]
    if past_pts:
        pts_ts, pts_vals = zip(*past_pts)
        ax.plot(pts_ts, pts_vals, color='red', linestyle='-', linewidth=1, alpha=0.4, zorder=5)
        ax.scatter(pts_ts, pts_vals, color='red', s=45, marker='x', label='Past Prediction', zorder=6)

    future_pts = [pt for pt in local_hist if pt[0] > last_ts]
    pred_val = 0
    if future_pts:
        future_ts, pred_val = future_pts[-1]
        ax.scatter(future_ts, pred_val, color='#f39c12', s=70, marker='x', linewidths=2, label='Next Target', zorder=5)

    display_now = now.replace(second=0, microsecond=0)
    start_view = display_now - timedelta(minutes=20)
    end_view = display_now + timedelta(minutes=10)
    
    visible_prices = local_df[(local_df['ts'] >= start_view) & (local_df['ts'] <= end_view)]['close'].tolist()
    visible_preds = [pt[1] for pt in local_hist if pt[0] >= start_view and pt[0] <= end_view]
    all_visible_values = visible_prices + visible_preds
    
    if all_visible_values:
        p_min, p_max = min(all_visible_values), max(all_visible_values)
        padding = (p_max - p_min) * 0.15 if p_max != p_min else 100
        y_min = ((p_min - padding) // 100) * 100
        y_max = ((p_max + padding) // 100 + 1) * 100
        ax.set_ylim(y_min, y_max)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.set_xlim(start_view, end_view)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))

    if future_pts:
        time_delta = future_ts - now
        seconds_remaining = int(max(0, time_delta.total_seconds()))
        countdown_str = f"{seconds_remaining // 60:02d}:{seconds_remaining % 60:02d}"
    else:
        countdown_str = "00:00"

    status_text = (
        f"SYMBOL  : {symbol_display}\n"
        f"TIME    : {now.strftime('%H:%M:%S')}\n"
        f"T-MINUS : {countdown_str}\n"
        f"PRICE   : ${last_price:,.2f}\n"
        f"PREDICT : ${pred_val:,.2f}\n"
        f"ACCURACY: {avg_acc:.2f}%\n"
        f"AVG ERR : ${avg_mae:.2f}"
    )
    
    ax.text(0.02, 0.95, status_text, transform=ax.transAxes, 
            bbox=dict(facecolor='black', alpha=0.85, edgecolor='#444444', boxstyle='round,pad=1'),
            fontsize=10, color='white', verticalalignment='top', family='monospace', weight='bold')

    fig.text(0.5, 0.95, "BITCOIN PRICE PREDICTOR", fontsize=18, fontweight='black', color='#ffffff', ha='center')
    fig.text(0.5, 0.905, f"Interval: 5 Minutes | Grid: $100 USD", fontsize=9, color='#888888', ha='center')
    
    ax.legend(loc='lower left', frameon=False, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

ani = FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
plt.show()