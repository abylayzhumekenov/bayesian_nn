% --- 1. Generate Synthetic Data ---
N = 100;   % Number of samples
n = 2;     % Number of input features
m = 1;     % Number of output dimensions (for regression)

% Generate random input data (N x n matrix)
X = rand(N, n);

% Generate random target values (N x m matrix)
Y = sin(X(:,1)) + cos(X(:,2)) + 0.1*randn(N, 1);  % Target is a noisy function

% --- 2. Initialize Neural Network Parameters ---
hidden_size = 10;  % Number of neurons in the hidden layer

% Xavier Initialization for the weights
W1 = randn(n, hidden_size) * sqrt(2 / (n + hidden_size));  % Input to hidden weights
b1 = zeros(1, hidden_size);  % Bias for the hidden layer

W2 = randn(hidden_size, m) * sqrt(2 / (hidden_size + m));  % Hidden to output weights
b2 = zeros(1, m);  % Bias for the output layer

% --- 3. Define Activation Function and Loss Function ---
% ReLU activation
relu = @(z) max(0, z);

% Mean Squared Error (MSE) loss
mse_loss = @(Y_pred, Y) mean((Y_pred - Y).^2);

% --- 4. Define Forward and Backward Pass ---
% Forward pass through the network
forward_pass = @(X, W1, b1, W2, b2) ...
    relu(X * W1 + repmat(b1, size(X, 1), 1)) * W2 + repmat(b2, size(X, 1), 1); 

% Backward pass (computing gradients using the chain rule)
backward_pass = @(X, Y, W1, b1, W2, b2, learning_rate) deal(dW1, db1, dW2, db2);

% --- 5. Train the Neural Network using Backpropagation ---
epochs = 1000;    % Number of training epochs
learning_rate = 0.01;  % Learning rate

% Initialize vectors to store weights, biases, and loss values
W1_vals = zeros(n, hidden_size, epochs);
b1_vals = zeros(1, hidden_size, epochs);
W2_vals = zeros(hidden_size, m, epochs);
b2_vals = zeros(1, m, epochs);
loss_vals = zeros(epochs, 1);

% Training Loop
for epoch = 1:epochs
    % Forward pass: Compute predicted Y
    Y_pred = forward_pass(X, W1, b1, W2, b2);
    
    % Compute loss
    loss = mse_loss(Y_pred, Y);
    
    % Store weights, biases, and loss at this epoch
    W1_vals(:,:,epoch) = W1;
    b1_vals(:,:,epoch) = b1;
    W2_vals(:,:,epoch) = W2;
    b2_vals(:,:,epoch) = b2;
    loss_vals(epoch) = loss;
    
    % --- Backpropagation ---
    % Compute gradients (using chain rule)
    dY = 2 * (Y_pred - Y) / N;  % Derivative of MSE loss w.r.t output
    dW2 = relu(X * W1 + repmat(b1, size(X, 1), 1))' * dY;  % Gradient of W2
    db2 = sum(dY, 1);  % Gradient of b2
    
    % Derivative of ReLU activation function
    dHidden = (dY * W2') .* (relu(X * W1 + repmat(b1, size(X, 1), 1)) > 0);  % Gradient of ReLU
    dW1 = X' * dHidden;  % Gradient of W1
    db1 = sum(dHidden, 1);  % Gradient of b1
    
    % Update weights using gradient descent
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
    
    % Display loss for every 100 epochs
    if mod(epoch, 100) == 0
        disp(['Epoch: ' num2str(epoch) ', Loss: ' num2str(loss)]);
    end
end

% --- 6. Plot Results ---
% Plot True Values vs Predicted Values
Y_pred_final = forward_pass(X, W1, b1, W2, b2);
figure;
subplot(1, 2, 1);
scatter(Y, Y_pred_final, 'r');
xlabel('True Values');
ylabel('Predicted Values');
title('True vs Predicted Values');
grid on;

% Plot Training Data with Regression Line
subplot(1, 2, 2);
plot3(X(:,1), X(:,2), Y, 'bo');  % True data points
hold on;
plot3(X(:,1), X(:,2), Y_pred_final, 'rx'); % Predicted points
xlabel('Feature 1');
ylabel('Feature 2');
zlabel('Target Value');
title('True vs Predicted Data');
grid on;

% --- 7. Plot Estimations of Weights and Biases Over Time ---
figure;

% Plot W1 evolution (weights from input to hidden layer)
subplot(3, 2, 1);
plot(squeeze(W1_vals(1, :, :))');
xlabel('Epoch');
ylabel('W1(1,:) Value');
title('Evolution of W1(1,:)');
grid on;

subplot(3, 2, 2);
plot(squeeze(W1_vals(2, :, :))');
xlabel('Epoch');
ylabel('W1(2,:) Value');
title('Evolution of W1(2,:)');
grid on;

% Plot W2 evolution (weights from hidden to output layer)
subplot(3, 2, 3);
plot(squeeze(W2_vals(1, :, :))');
xlabel('Epoch');
ylabel('W2(1,:) Value');
title('Evolution of W2(1,:)');
grid on;

subplot(3, 2, 4);
plot(squeeze(W2_vals(2, :, :))');
xlabel('Epoch');
ylabel('W2(2,:) Value');
title('Evolution of W2(2,:)');
grid on;

% Plot Bias evolution
subplot(3, 2, 5);
plot(squeeze(b1_vals(1, :, :))');
xlabel('Epoch');
ylabel('b1 Value');
title('Evolution of b1');
grid on;

subplot(3, 2, 6);
plot(squeeze(b2_vals(1, :, :))');
xlabel('Epoch');
ylabel('b2 Value');
title('Evolution of b2');
grid on;

% --- 8. Plot Loss Evolution ---
figure;
plot(1:epochs, loss_vals);
xlabel('Epoch');
ylabel('Loss (MSE)');
title('Loss Evolution Over Time');
grid on;
