import numpy as np
import json

# Model parameter handling
def store_model_parameters(model, filename):
    model_data = {key: value.tolist() for key, value in model.items()}
    with open(f'{filename}.json', 'w') as file:
        json.dump(model_data, file)

def retrieve_model_parameters(filename):
    with open(f'{filename}.json', 'r') as file:
        model_data = json.load(file)
    return {key: np.array(value) for key, value in model_data.items()}

# Activation functions and their derivatives
def relu_activation(x):
    return np.maximum(x, 0)

def softmax_activation(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return x * (1 - x)

# Loss computation
def calculate_loss(true_y, predicted_y, model, regularization_factor):
    reg_loss = regularization_factor / 2 * sum(np.sum(weights ** 2) for weights in [model['W1'], model['W2'], model['W3']])
    log_loss = -np.mean(np.sum(true_y * np.log(predicted_y + 1e-12), axis=1))
    return log_loss + reg_loss

def setup_model(input_size, hidden_size, output_size):
    model = {
        'W1': np.random.randn(input_size, hidden_size) * 0.01,
        'b1': np.zeros(hidden_size),
        'W2': np.random.randn(hidden_size, hidden_size) * 0.01,
        'b2': np.zeros(hidden_size),
        'W3': np.random.randn(hidden_size, output_size) * 0.01,
        'b3': np.zeros(output_size)
    }
    return model

# Forward and backward propagation
def model_forward(model, X, activation='relu'):
    activations = {}
    W1, b1, W2, b2, W3, b3 = model.values()

    activations['z1'] = X @ W1 + b1
    activations['a1'] = relu_activation(activations['z1']) if activation == 'relu' else sigmoid_activation(activations['z1'])

    activations['z2'] = activations['a1'] @ W2 + b2
    activations['a2'] = relu_activation(activations['z2']) if activation == 'relu' else sigmoid_activation(activations['z2'])

    activations['z3'] = activations['a2'] @ W3 + b3
    activations['a3'] = softmax_activation(activations['z3'])

    return activations['a3'], activations

def model_backward(model, activations, X, y, y_hat, reg_strength, activation='relu'):
    gradients = {}
    W1, W2, W3 = model['W1'], model['W2'], model['W3']
    a1, a2 = activations['a1'], activations['a2']

    error = y_hat - y
    gradients['dW3'] = a2.T @ error + reg_strength * W3
    gradients['db3'] = np.sum(error, axis=0)

    if activation == 'relu':
        error = (error @ W3.T) * (a2 > 0)
    else:
        error = (error @ W3.T) * derivative_sigmoid(a2)
    
    gradients['dW2'] = a1.T @ error + reg_strength * W2
    gradients['db2'] = np.sum(error, axis=0)

    if activation == 'relu':
        error = (error @ W2.T) * (a1 > 0)
    else:
        error = (error @ W2.T) * derivative_sigmoid(a1)

    gradients['dW1'] = X.T @ error + reg_strength * W1
    gradients['db1'] = np.sum(error, axis=0)

    return gradients

def update_parameters(model, gradients, learn_rate):
    for key in model:
        model[key] -= learn_rate * gradients['d' + key]
    return model

def calculate_accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

def encode_one_hot(labels, num_classes):
    one_hot_labels = np.eye(num_classes)[labels.flatten()]
    return one_hot_labels