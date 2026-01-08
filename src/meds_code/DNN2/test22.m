% --- 1. Generate Synthetic Data ---
N = 100;   % Number of samples
n = 2;     % Number of input features
m = 1;     % Number of output dimensions (for regression)

X = rand(N, n);  % Generate random input data
w_true = [2; -3];  % True weights for linear regression
Y = X * w_true + 0.1 * randn(N, 1);  % Linear function with noise

% --- 2. Initialize Neural Network Parameters ---
hidden_size = 10;  % Hidden layer neurons

% Initialize posterior mean and variance for weights
A1 = randn(n, hidden_size) * sqrt(2 / (n + hidden_size));
B1 = ones(n, hidden_size);  % Variance (uncertainty measure)

A2 = randn(hidden_size, m) * sqrt(2 / (hidden_size + m));
B2 = ones(hidden_size, m);  % Variance (uncertainty measure)

b1 = zeros(1, hidden_size); 
b2 = zeros(1, m);

% --- 3. Activation and Loss Functions ---
relu = @(z) max(0, z);
mse_loss = @(Y_pred, Y) mean((Y_pred - Y).^2);

% --- 4. Train Using Martingale Posterior Updates ---
epochs = 1000;
learning_rate = 0.01;
alpha = 0.9;  % Confidence-weighted update factor

A1_vals = zeros(n, hidden_size, epochs);
A2_vals = zeros(hidden_size, m, epochs);
b1_vals = zeros(1, hidden_size, epochs);
b2_vals = zeros(1, m, epochs);
loss_vals = zeros(epochs, 1);

for epoch = 1:epochs
    % Forward pass
    H = relu(X * A1 + repmat(b1, N, 1));
    Y_pred = H * A2 + repmat(b2, N, 1);
    
    % Compute loss
    loss = mse_loss(Y_pred, Y);
    
    % Store parameter evolution
    A1_vals(:,:,epoch) = A1;
    A2_vals(:,:,epoch) = A2;
    b1_vals(:,:,epoch) = b1;
    b2_vals(:,:,epoch) = b2;
    loss_vals(epoch) = loss;
    
    % Compute gradients
    dY = 2 * (Y_pred - Y) / N;
    dA2 = H' * dY;
    db2 = sum(dY, 1);
    
    dHidden = (dY * A2') .* (H > 0);
    dA1 = X' * dHidden;
    db1 = sum(dHidden, 1);
    
    % Bayesian Posterior Update (Martingale)
    B1 = alpha * B1 + (1 - alpha) * abs(dA1);
    B2 = alpha * B2 + (1 - alpha) * abs(dA2);
    
    A1 = A1 - learning_rate * dA1 ./ (B1 + 1e-6);
    A2 = A2 - learning_rate * dA2 ./ (B2 + 1e-6);
    b1 = b1 - learning_rate * db1;
    b2 = b2 - learning_rate * db2;
    
    if mod(epoch, 100) == 0
        disp(['Epoch: ' num2str(epoch) ', Loss: ' num2str(loss)]);
    end
end

% --- 5. Plot Results ---
figure;
plot(1:epochs, loss_vals);
xlabel('Epoch');
ylabel('Loss (MSE)');
title('Loss Evolution Over Time');
grid on;

figure;
subplot(2,1,1);
plot(squeeze(mean(A1_vals, [1,2])));
xlabel('Epoch');
ylabel('Mean A1 Values');
title('Evolution of Posterior Mean A1');
grid on;

subplot(2,1,2);
plot(squeeze(mean(A2_vals, [1,2])));
xlabel('Epoch');
ylabel('Mean A2 Values');
title('Evolution of Posterior Mean A2');
grid on;