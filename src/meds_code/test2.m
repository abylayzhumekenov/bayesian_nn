% Bayesian Online Learning for Linear Regression with Martingale Posterior Updates
clear; clc; rng(42);

% Generate synthetic data (y = X*beta + noise)
n = 100;  % Number of data points
d = 2;    % Number of features
X = [ones(n,1), randn(n, d-1)]; % Feature matrix with bias term
beta_true = [2; -1]; % True parameters
sigma = 0.5; % Noise standard deviation
Y = X * beta_true + sigma * randn(n,1); % Response variable

% Prior (Gaussian)
mu_prior = zeros(d,1);       % Prior mean
Sigma_prior = eye(d);        % Prior covariance

% Initialize posterior
mu_posterior = mu_prior;
Sigma_posterior = Sigma_prior;

% Step-size schedule
epsilon = @(m) 1/(m+1); 

% Sequential Bayesian Updating (Martingale Posterior)
beta_estimates = zeros(d, n); % Store estimates for plotting

for m = 1:n
    X_m = X(m, :);  % New data point
    Y_m = Y(m);     % New observation
    
    % Compute score function s(X, Y, beta)
    residual = Y_m - X_m * mu_posterior; % Prediction error
    score = (1/sigma^2) * X_m' * residual; % Score function
    
    % Update parameter using martingale posterior rule
    mu_posterior = mu_posterior + epsilon(m) * score;
    
    % Store estimate for visualization
    beta_estimates(:, m) = mu_posterior;
end

% Plot posterior evolution
figure;
subplot(2,1,1);
plot(1:n, beta_estimates(1, :), 'b', 'LineWidth', 2); hold on;
yline(beta_true(1), 'r--', 'LineWidth', 2);
xlabel('Iteration'); ylabel('\beta_1 estimate');
title('Evolution of \beta_1');

subplot(2,1,2);
plot(1:n, beta_estimates(2, :), 'b', 'LineWidth', 2); hold on;
yline(beta_true(2), 'r--', 'LineWidth', 2);
xlabel('Iteration'); ylabel('\beta_2 estimate');
title('Evolution of \beta_2');

sgtitle('Martingale Posterior Updates for Bayesian Linear Regression');
