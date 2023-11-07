from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with your secret key
socketio = SocketIO(app)

# Define a model with three LSTM layers and one dense layer
model = Sequential([
    LSTM(units=5, return_sequences=True, input_shape=(2, 1)),
    LSTM(units=5, return_sequences=True),
    LSTM(units=5, return_sequences=True),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('get_initial_model')
def send_initial_model():
    initial_model_weights = model.get_weights()
    for i, layer_weights in enumerate(initial_model_weights):
        layer_weights_dict = {
            'layerIndex': i,
            'weights': [weight.tolist() for weight in layer_weights]
        }
        emit('receive_initial_model', layer_weights_dict)

# The rest of your code for handling training and emitting model updates remains the same

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
