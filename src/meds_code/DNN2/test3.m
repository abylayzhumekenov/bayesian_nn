%% Martingale Posterior for Deep Neural Network (DNN)
clear;
close all;
clc;

%% 1. Generate Synthetic Data
N = 100;   % Number of samples
n = 10;    % Number of input features
m = 1;     % Output dimension

X = dlarray(rand(N, n));  % Input data as dlarray (No labels needed)
Y = dlarray(sum(sin(X), 2) + 0.1 * randn(N, 1));  % Noisy target

%% 2. Initialize Neural Network Parameters
hidden_size = 20;  % Hidden layer size

W1 = dlarray(randn(n, hidden_size) * sqrt(2 / (n + hidden_size)));
b1 = dlarray(zeros(1, hidden_size));
W2 = dlarray(randn(hidden_size, m) * sqrt(2 / (hidden_size + m)));
b2 = dlarray(zeros(1, m));

%% 3. Define ReLU Activation Function
relu = @(z) max(z, 0);  % ReLU function

%% 4. Train Network with Backpropagation (Pre-training Step)
epochs = 1000;
learning_rate = 0.01;
loss_values = zeros(epochs, 1);
W1_values = [];  % To store W1 values over epochs
W2_values = [];  % To store W2 values over epochs
b1_values = [];  % To store b1 values over epochs
b2_values = [];  % To store b2 values over epochs

for epoch = 1:epochs
    % Compute loss and gradients using automatic differentiation
    [loss, gradients] = dlfeval(@modelGradients, X, Y, W1, b1, W2, b2, relu);
    loss_values(epoch) = extractdata(loss);
    
    % Update parameters
    W1 = W1 - learning_rate * gradients.dW1;
    b1 = b1 - learning_rate * gradients.db1;
    W2 = W2 - learning_rate * gradients.dW2;
    b2 = b2 - learning_rate * gradients.db2;

    % Save weights and biases values over time
    W1_values = [W1_values, extractdata(W1)];
    W2_values = [W2_values, extractdata(W2)];
    b1_values = [b1_values, extractdata(b1)];
    b2_values = [b2_values, extractdata(b2)];
end

%% 5. Martingale Posterior Update Using Score Function
epsilon_0 = 0.1;

for m = 1:N
    % Extract single data point
    X_m = X(m, :);
    Y_m = Y(m, :);
    
    % Compute gradients for single data point
    [loss, gradients] = dlfeval(@modelGradients, X_m, Y_m, W1, b1, W2, b2, relu);
    
    % Adaptive learning rate
    epsilon_m = epsilon_0 / (m + 1);
    
    % Update parameters
    W1 = W1 + epsilon_m * gradients.dW1;
    b1 = b1 + epsilon_m * gradients.db1;
    W2 = W2 + epsilon_m * gradients.dW2;
    b2 = b2 + epsilon_m * gradients.db2;
end

%% 6. Plot Results
plot_results(X, Y, W1, b1, W2, b2, loss_values, W1_values, W2_values, b1_values, b2_values);

%% Function Definitions
function [loss, gradients] = modelGradients(X, Y, W1, b1, W2, b2, relu)
    % Forward pass
    hidden_layer = relu(X * W1 + b1);
    Y_pred = hidden_layer * W2 + b2;
    
    % Compute loss
    loss = mse_loss(Y_pred, Y);
    
    % Compute gradients
    gradients.dW1 = dlgradient(loss, W1);
    gradients.db1 = dlgradient(loss, b1);
    gradients.dW2 = dlgradient(loss, W2);
    gradients.db2 = dlgradient(loss, b2);
end

function loss = mse_loss(Y_pred, Y)
    loss = mean((Y_pred - Y).^2, 'all');  % Ensure scalar loss
end

function plot_results(X, Y, W1, b1, W2, b2, loss_values, W1_values, W2_values, b1_values, b2_values)
    Y_pred = predict(X, W1, b1, W2, b2, @(z) max(0, z));

    % True vs Predicted Plot
    figure;
    subplot(1, 2, 1);
    scatter(extractdata(Y), extractdata(Y_pred), 'r');
    xlabel('True Values'); ylabel('Predicted Values');
    title('True vs Predicted Values'); grid on;

    % Loss Evolution Plot
    subplot(1, 2, 2);
    plot(loss_values, '-b');
    xlabel('Epochs'); ylabel('Loss');
    title('Loss Evolution'); grid on;

    % Plot the evolution of weights and biases
    figure;
    subplot(2, 2, 1);
    plot(W1_values');
    xlabel('Epochs'); ylabel('Weights W1');
    title('Evolution of W1'); grid on;
    
    subplot(2, 2, 2);
    plot(W2_values');
    xlabel('Epochs'); ylabel('Weights W2');
    title('Evolution of W2'); grid on;
    
    subplot(2, 2, 3);
    plot(b1_values');
    xlabel('Epochs'); ylabel('Biases b1');
    title('Evolution of b1'); grid on;
    
    subplot(2, 2, 4);
    plot(b2_values');
    xlabel('Epochs'); ylabel('Biases b2');
    title('Evolution of b2'); grid on;
end

function Y_pred = predict(X, W1, b1, W2, b2, relu)
    hidden_layer = relu(X * W1 + b1);
    Y_pred = hidden_layer * W2 + b2;
end
