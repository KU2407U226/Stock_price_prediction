📈 Stock Price Predictor
👨‍💻 Group Members
Swar Prajapati (KU2407U787)

Digvijaysinh Vala (KU2407U226)

Krrish Chudasama (KU2407U120)

🎯 Objective of the Project:

Build a web application that predicts future stock prices based on historical data using Machine Learning models and visualize important moving averages like 50-day, 100-day, and 200-day averages.

🛠️ Tools and Libraries Used:

Python 3

Flask

Keras

NumPy

Pandas

yfinance

scikit-learn

Matplotlib

Gunicorn (for deployment)

🔗 Data Source:
Stock data fetched live from Yahoo Finance using the yfinance Python library.

⚙️ Execution Steps:

1. Clone the repository:

git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor

2. Install dependencies:

pip install -r requirements.txt

3. Add your trained model file (newmodel.keras) inside the project folder.

4. Run the application:

python app.py

5. Open your browser and go to:

http://localhost:5000/

📊 Summary of Results:

-Stock data was successfully retrieved.

-Moving Averages (50, 100, 200 days) plotted with actual closing prices.

-Future stock prices predicted and compared with actual prices.

-A simple, easy-to-use web interface was created.

🚧 Challenges Faced:

-Handling invalid stock symbols

-Scaling data correctly for predictions.

