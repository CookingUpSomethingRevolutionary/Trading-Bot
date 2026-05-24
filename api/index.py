from flask import Flask, render_template, request, jsonify
from datetime import datetime

app = Flask(__name__, template_folder='../templates')

# In-memory storage cache for serverless tracking display
trade_history = []

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/api/log', methods=['POST'])
def log_trade():
    data = request.json
    trade_item = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": data.get("symbol", "SPY"),
        "action": data.get("action"),
        "qty": data.get("qty"),
        "price": data.get("price")
    }
    trade_history.insert(0, trade_item) # Keep newest at the top
    return jsonify({"status": "success", "logged": trade_item}), 200

@app.route('/api/trades', methods=['GET'])
def get_trades():
    return jsonify(trade_history)

# Adaptor layer required for Vercel WSGI
def handler(environ, start_response):
    return app(environ, start_response)