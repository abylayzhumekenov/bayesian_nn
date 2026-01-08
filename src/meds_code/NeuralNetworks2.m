%% MATLAB Implementation of Martingale Posterior Updates for Neural Networks
clear; clc; close all;

%% Parameters
input_dim = 3; % Number of input features
hidden_dim = 10; % Number of hidden units
output_dim = 1; % Number of output neurons (for regression)
n_samples = 1000; % Number of observations

%% Initialize Parameters
A1 = randn(hidden_dim, input_dim) * 0.1; % Weight matrix for first layer
b1 = randn(hidden_dim, 1) * 0.1; % Bias for first layer
A2 = randn(output_dim, hidden_dim) * 0.1; % Weight matrix for output layer
b2 = randn(output_dim, 1) * 0.1; % Bias for output layer

%% Generate synthetic data
X = randn(n_samples, input_dim); % Inputs
y = randn(n_samples, output_dim); % Targets (for simplicity)

%% Step size function
epsilon_n = @(n) 1 / (n + 10);

%% Store parameter estimates for visualization
A1_history = zeros(hidden_dim, input_dim, n_samples);
b1_history = zeros(hidden_dim, n_samples);
A2_history = zeros(output_dim, hidden_dim, n_samples);
b2_history = zeros(output_dim, n_samples);
loss_history = zeros(1, n_samples);

%% Sequential Parameter Updates
for n = 1:n_samples
    xn = reshape(X(n, :), [input_dim, 1]); % Current input sample
    yn = y(n); % Current output sample
    eps_n = epsilon_n(n); % Adaptive step size
    
    % Forward pass
    z1 = A1 * xn + b1;
    a1 = max(0, z1); % ReLU activation
    y_pred = A2 * a1 + b2;
    
    % Compute loss (Mean Squared Error)
    loss = (yn - y_pred)^2;
    loss_history(n) = loss;
    
    % Compute gradients
    dL_dy = -2 * (yn - y_pred);
    dL_dA2 = dL_dy * a1';
    dL_db2 = dL_dy;
    
    dL_da1 = A2' * dL_dy;
    dL_dz1 = dL_da1 .* (z1 > 0);
    dL_dA1 = dL_dz1 * xn';
    dL_db1 = dL_dz1;
    
    % Clamp gradients to prevent NaN explosion
    dL_dA1 = min(max(dL_dA1, -10), 10);
    dL_db1 = min(max(dL_db1, -10), 10);
    dL_dA2 = min(max(dL_dA2, -10), 10);
    dL_db2 = min(max(dL_db2, -10), 10);

    % Update Parameters
    A1 = A1 - eps_n * dL_dA1;
    b1 = b1 - eps_n * dL_db1;
    A2 = A2 - eps_n * dL_dA2;
    b2 = b2 - eps_n * dL_db2;

    % Store parameters for visualization
    A1_history(:, :, n) = A1;
    b1_history(:, n) = b1;
    A2_history(:, :, n) = A2;
    b2_history(:, n) = b2;
end

%% Display final parameter estimates
disp('Final Estimated Parameters:');
disp('A1 ='); disp(A1);
disp('b1 ='); disp(b1);
disp('A2 ='); disp(A2);
disp('b2 ='); disp(b2);

%% Plot A1 Evolution (Evolution of Weights in A1)
figure('Position', [100, 100, 800, 900]); 
for i = 1:hidden_dim
    for j = 1:input_dim
        subplot(hidden_dim, input_dim, (i-1)*input_dim + j);
        plot(1:n_samples, squeeze(A1_history(i, j, :)), 'LineWidth', 1.5);
        xlabel('Iterations', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel(['A1(' num2str(i) ',' num2str(j) ')'], 'FontSize', 12, 'FontWeight', 'bold');
        title(['Evolution of A1(' num2str(i) ',' num2str(j) ')'], 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
    end
end

%% Plot b1 Evolution (Bias terms in the hidden layer)
figure('Position', [100, 100, 800, 600]);
for i = 1:hidden_dim
    subplot(hidden_dim, 1, i);
    plot(1:n_samples, squeeze(b1_history(i, :)), 'LineWidth', 1.5);
    xlabel('Iterations', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(['b1_' num2str(i)], 'FontSize', 12, 'FontWeight', 'bold');
    title(['Evolution of b1_' num2str(i)], 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
end

%% Plot A2 Evolution (Evolution of Weights in A2)
figure('Position', [100, 100, 800, 900]);
for i = 1:output_dim
    for j = 1:hidden_dim
        subplot(output_dim, hidden_dim, (i-1)*hidden_dim + j);
        plot(1:n_samples, squeeze(A2_history(i, j, :)), 'LineWidth', 1.5);
        xlabel('Iterations', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel(['A2(' num2str(i) ',' num2str(j) ')'], 'FontSize', 12, 'FontWeight', 'bold');
        title(['Evolution of A2(' num2str(i) ',' num2str(j) ')'], 'FontSize', 14, 'FontWeight', 'bold');
        grid on;
    end
end

%% Plot b2 Evolution (Bias terms for output layer)
figure('Position', [100, 100, 800, 600]);
for i = 1:output_dim
    subplot(output_dim, 1, i);
    plot(1:n_samples, squeeze(b2_history(i, :)), 'LineWidth', 1.5);
    xlabel('Iterations', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(['b2_' num2str(i)], 'FontSize', 12, 'FontWeight', 'bold');
    title(['Evolution of b2_' num2str(i)], 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
end
