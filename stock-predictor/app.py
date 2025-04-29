from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import base64
from io import BytesIO
import os

app = Flask(__name__)

# Load your model (use your actual path)
model =load_model('newmodel.keras')

def create_plot():
    """Helper function to create matplotlib plots"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    plots = {}
    prediction_data = None
    
    if request.method == 'POST':
        stock = request.form['stock'].upper()
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        try:
            # Download data
            data = yf.download(stock, start_date, end_date)
            
            if data.empty:
                raise ValueError("No data found for this stock symbol")
            
            # Prepare data for model
            data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
            data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
            
            scaler = MinMaxScaler(feature_range=(0,1))
            pas_100_days = data_train.tail(100)
            data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
            data_test_scale = scaler.fit_transform(data_test)

            # Generate MA plots
            ma_50_days = data.Close.rolling(50).mean()
            ma_100_days = data.Close.rolling(100).mean()
            ma_200_days = data.Close.rolling(200).mean()
            
            # Plot 1: Price vs MA50
            plt.figure(figsize=(10,6))
            plt.plot(ma_50_days, 'r', label='50-Day MA')
            plt.plot(data.Close, 'g', label='Closing Price')
            plt.title(f'{stock} Price vs 50-Day Moving Average')
            plt.legend()
            plots['ma50'] = create_plot()
            plt.close()
            
            # Plot 2: Price vs MA50 vs MA100
            plt.figure(figsize=(10,6))
            plt.plot(ma_50_days, 'r', label='50-Day MA')
            plt.plot(ma_100_days, 'b', label='100-Day MA')
            plt.plot(data.Close, 'g', label='Closing Price')
            plt.title(f'{stock} Price vs Moving Averages')
            plt.legend()
            plots['ma100'] = create_plot()
            plt.close()
            
            # Plot 3: Price vs MA100 vs MA200
            plt.figure(figsize=(10,6))
            plt.plot(ma_100_days, 'r', label='100-Day MA')
            plt.plot(ma_200_days, 'b', label='200-Day MA')
            plt.plot(data.Close, 'g', label='Closing Price')
            plt.title(f'{stock} Price vs Long-Term Moving Averages')
            plt.legend()
            plots['ma200'] = create_plot()
            plt.close()
            
            # Prepare prediction data
            x, y = [], []
            for i in range(100, data_test_scale.shape[0]):
                x.append(data_test_scale[i-100:i])
                y.append(data_test_scale[i,0])
            
            x, y = np.array(x), np.array(y)
            predict = model.predict(x)
            
            scale = 1/scaler.scale_
            predict = predict * scale
            y = y * scale
            
            # Plot 4: Original vs Predicted
            plt.figure(figsize=(10,6))
            plt.plot(predict, 'r', label='Predicted Price')
            plt.plot(y, 'g', label='Actual Price')
            plt.title(f'{stock} Price Prediction vs Actual')
            plt.legend()
            plots['prediction'] = create_plot()
            plt.close()
            
            prediction_data = {
                'stock': stock,
                'start_date': start_date,
                'end_date': end_date,
                'data': data.tail(10).to_html(classes='data-table'),
                'predictions': list(predict.flatten())[-5:]  # Last 5 predictions
            }
            
        except Exception as e:
            prediction_data = {'error': str(e)}
    
    return render_template('index.html', plots=plots, prediction=prediction_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
