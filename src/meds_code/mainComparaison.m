clear; clc; close all;

% Parameters
p = 3; % Number of predictors
n_samples = 1000; % Number of observations

% True parameters (for simulation)
beta_true = [1; -2; 3]; % True regression coefficients
sigma2_true = 1; % True noise variance
mu_X_true = [0; 0; 0]; % True mean of predictors
Sigma_X_true = eye(p); % True covariance matrix of predictors

% Generate synthetic data
X = mvnrnd(mu_X_true', Sigma_X_true, n_samples); % Generate predictors from normal distribution
y = X * beta_true + sqrt(sigma2_true) * randn(n_samples, 1); % Generate responses

% Step 1: Compute Exact Posterior Mean and Covariance without Priors
X_full = X(1:end, :); % All X data
y_full = y(1:end);    % All y data

% Exact posterior mean for beta (OLS solution)
beta_exact = (X_full' * X_full) \ (X_full' * y_full); 

% Exact posterior covariance for beta
Sigma_exact = sigma2_true * inv(X_full' * X_full);

% Exact posterior variance for sigma^2
sigma2_exact = (y_full - X_full * beta_exact)' * (y_full - X_full * beta_exact) / n_samples;

% Step 2: Initialize parameters for sequential updates (Martingale Posterior)
beta = zeros(p, 1);  % Initial estimate for beta
sigma2 = 1;  % Initial estimate for sigma^2
mu_X = zeros(p, 1);  % Initial estimate for mu_X
Sigma_X = eye(p); % Initial estimate for Sigma_X

% Step size function
epsilon_n = @(n) 1/(n+1);  % Adaptive step size

% Store parameter estimates for plotting
beta_history = zeros(p, n_samples);
sigma2_history = zeros(1, n_samples);
mu_X_history = zeros(p, n_samples);
Sigma_X_history = zeros(p, p, n_samples);

%% Step 3: Sequential updating (Martingale Posterior)
for n = 1:n_samples
    xn = X(n, :)';  % Current predictor sample (column vector)
    yn = y(n);      % Current response sample
    eps_n = epsilon_n(n);  % Adaptive step size
    
    % Update beta (Martingale Posterior)
    beta = beta + eps_n * (xn * (yn - xn' * beta));
    
    % Update sigma^2 (Martingale Posterior)
    sigma2 = sigma2 + eps_n * ((yn - xn' * beta)^2 - sigma2);
    
    % Update mu_X
    mu_X = mu_X + eps_n * (xn - mu_X);
    
    % Update Sigma_X
    Sigma_X = Sigma_X + eps_n * ((xn - mu_X) * (xn - mu_X)' - Sigma_X);
    
    % Enforce symmetry in Sigma_X to prevent numerical drift
    Sigma_X = 0.5 * (Sigma_X + Sigma_X');
    
    % Store history for visualization
    beta_history(:, n) = beta;
    sigma2_history(n) = sigma2;
    mu_X_history(:, n) = mu_X;
    Sigma_X_history(:, :, n) = Sigma_X;
    
    % Compute exact posterior mean and covariance at step n
    X_n = X(1:n, :);
    y_n = y(1:n);
    
    % Exact posterior mean for beta
    beta_exact_n = (X_n' * X_n) \ (X_n' * y_n); 
    
    % Exact posterior covariance for beta
    Sigma_exact_n = sigma2_true * inv(X_n' * X_n);
    
    % Exact posterior variance for sigma^2
    sigma2_exact_n = (y_n - X_n * beta_exact_n)' * (y_n - X_n * beta_exact_n) / n;
    
    % Compare estimates with true analytical posterior
    if mod(n, 100) == 0  % Display progress every 100 iterations
        fprintf('Iteration %d\n', n);
        fprintf('Martingale Posterior beta:\n'); disp(beta);
        fprintf('Exact Posterior beta:\n'); disp(beta_exact_n);
        fprintf('Martingale Posterior sigma^2: %f\n', sigma2);
        fprintf('Exact Posterior sigma^2: %f\n', sigma2_exact_n);
        fprintf('Martingale Posterior Covariance Matrix:\n'); disp(Sigma_X);
        fprintf('Exact Posterior Covariance Matrix:\n'); disp(Sigma_exact_n);
        fprintf('--------------------------------------\n');
    end
end

%% Plot Results
% Plot estimation history of beta
figure;
for i = 1:p
    subplot(p, 1, i);
    plot(1:n_samples, beta_history(i, :), 'b-', 'LineWidth', 1.5); hold on;
    yline(beta_true(i), 'r--', 'LineWidth', 1.5); % True value in red dashed line
    xlabel('Iterations');
    ylabel(['\beta_' num2str(i)]);
    title(['Estimation of \beta_' num2str(i)]);
    legend('Estimated', 'True');
    grid on;
end

% Plot estimation history of sigma^2
figure;
plot(1:n_samples, sigma2_history, 'b-', 'LineWidth', 1.5); hold on;
yline(sigma2_true, 'r--', 'LineWidth', 1.5);
xlabel('Iterations');
ylabel('\sigma^2');
title('Estimation of \sigma^2');
legend('Estimated', 'True');
grid on;

% Plot estimation history of elements of Sigma_X
figure;
idx = 1;
for i = 1:p
    for j = i:p  % Only upper triangular part including diagonal
        subplot(p, p, idx);
        plot(1:n_samples, squeeze(Sigma_X_history(i, j, :)), 'b-', 'LineWidth', 1.5); hold on;
        yline(Sigma_X_true(i, j), 'r--', 'LineWidth', 1.5);
        xlabel('Iterations');
        ylabel(['\Sigma_{' num2str(i) ',' num2str(j) '}']);
        title(['Estimation of \Sigma_{' num2str(i) ',' num2str(j) '}']);
        legend('Estimated', 'True');
        grid on;
        idx = idx + 1;
    end
end
