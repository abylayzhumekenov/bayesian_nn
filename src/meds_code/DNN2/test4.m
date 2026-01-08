clear;
close all;
clc;

%% 1. Generate Synthetic Data
N = 100;   % Number of samples
n = 10;    % Number of input features (higher dimension)
m = 1;     % Output dimension (regression)

X = rand(N, n);  % Random input data
Y = sum(sin(X), 2) + 0.1 * randn(N, 1);  % Noisy target function

%% 2. Initialize Neural Network Parameters
hidden_size = 20;  % Hidden layer size

% Initialize parameters as dlarray with 'CB' labels (Channel, Batch)
W1 = dlarray(randn(n, hidden_size) * sqrt(2 / (n + hidden_size)), 'CB');
b1 = dlarray(zeros(1, hidden_size), 'CB');
W2 = dlarray(randn(hidden_size, m) * sqrt(2 / (hidden_size + m)), 'CB');
b2 = dlarray(zeros(1, m), 'CB');

%% 3. Define ReLU Activation Function
relu = @(z) max(0, z);  % ReLU function

%% 4. Train Network with Backpropagation (Pre-training Step)
epochs = 1000;
learning_rate = 0.01;
loss_values = zeros(epochs, 1);

for epoch = 1:epochs
    % Forward pass: Perform matrix multiplication using extractdata for numeric operations
    W1_data = extractdata(W1);  % Extract numeric data from dlarray W1
    b1_data = extractdata(b1);  % Extract numeric data from dlarray b1
    W2_data = extractdata(W2);  % Extract numeric data from dlarray W2
    b2_data = extractdata(b2);  % Extract numeric data from dlarray b2
    
    % Compute activations and predictions using numeric matrices (non-dlarray)
    hidden_layer = relu(X * W1_data + b1_data);  % Apply ReLU activation
    Y_pred = hidden_layer * W2_data + b2_data;   % Output prediction

    % Compute loss
    loss = mse_loss(Y_pred, Y);
    loss_values(epoch) = loss;
    
    % Compute gradients using dlgradient
    dY = 2 * (Y_pred - Y) / N;
    hidden_layer_activation = relu(X * W1_data + b1_data);  % Apply ReLU activation

    % Backpropagate through the network
    dHidden = (dY * W2_data') .* (hidden_layer_activation > 0);  % Apply ReLU derivative
    dW2 = hidden_layer' * dY;  % Compute gradient for W2
    db2 = sum(dY, 1);  % Compute gradient for b2
    dW1 = X' * dHidden;  % Compute gradient for W1
    db1 = sum(dHidden, 1);  % Compute gradient for b1
    
    % Convert gradients to dlarray before updating
    dW1 = dlarray(dW1, 'CB');
    db1 = dlarray(db1, 'CB');
    dW2 = dlarray(dW2, 'CB');
    db2 = dlarray(db2, 'CB');
    
    % Update parameters using element-wise operations
    W1 = W1 - learning_rate .* dW1;
    b1 = b1 - learning_rate .* db1;
    W2 = W2 - learning_rate .* dW2;
    b2 = b2 - learning_rate .* db2;
end

%% 5. Plot Results
plot_results(X, Y, W1, b1, W2, b2, loss_values);

%% Function Definitions
function Y_pred = forward_pass(X, W1, b1, W2, b2, relu)
    % Forward pass: Use extractdata for dlarray handling
    W1_data = extractdata(W1);
    b1_data = extractdata(b1);
    W2_data = extractdata(W2);
    b2_data = extractdata(b2);

    hidden_layer = relu(X * W1_data + b1_data);  % Apply ReLU
    Y_pred = hidden_layer * W2_data + b2_data;
end

function loss = mse_loss(Y_pred, Y)
    loss = mean((Y_pred - Y).^2, 'all');  % Ensure loss is a scalar
end

function plot_results(X, Y, W1, b1, W2, b2, loss_values)
    % Make predictions with the final model
    Y_pred = forward_pass(X, W1, b1, W2, b2, @(z) max(0, z));

    % Ensure the predictions and true values are extracted from dlarray
    if isa(Y_pred, 'dlarray')
        Y_pred_values = extractdata(Y_pred);  % Convert dlarray to numeric array
    else
        Y_pred_values = Y_pred;  % If already a regular numeric array, skip extractdata
    end

    % Ensure that Y is a regular numeric array
    Y_values = Y;  % Y is already a regular numeric array, no need for extractdata()

    % True vs Predicted Plot
    figure;
    subplot(1, 2, 1);
    scatter(Y_values, Y_pred_values, 'r');  % Scatter plot of true vs predicted values
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
    plot(1:size(W1, 1), extractdata(W1), '-r');
    xlabel('Epochs'); ylabel('Weight Values');
    title('Evolution of W1'); grid on;

    subplot(2, 2, 2);
    plot(1:size(W2, 1), extractdata(W2), '-g');
    xlabel('Epochs'); ylabel('Weight Values');
    title('Evolution of W2'); grid on;

    subplot(2, 2, 3);
    plot(1:size(b1, 1), extractdata(b1), '-b');
    xlabel('Epochs'); ylabel('Bias Values');
    title('Evolution of b1'); grid on;

    subplot(2, 2, 4);
    plot(1:size(b2, 1), extractdata(b2), '-m');
    xlabel('Epochs'); ylabel('Bias Values');
    title('Evolution of b2'); grid on;
end
