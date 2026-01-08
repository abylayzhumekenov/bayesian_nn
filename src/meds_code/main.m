% MATLAB Implementation of Martingale Posterior Updates for Linear Regression
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
X = mvnrnd(mu_X_true, Sigma_X_true, n_samples); % Generate predictors from normal distribution
y = X * beta_true + sqrt(sigma2_true) * randn(n_samples, 1); % Generate responses

%% Step 1: Compute Maximum Likelihood Estimates (MLE) for initialization
beta = (X' * X) \ (X' * y);  % MLE for beta (OLS solution)
sigma2 = mean((y - X * beta).^2);  % MLE for sigma^2 (variance of residuals)
mu_X = mean(X, 1)';  % MLE for mu_X (column vector of means)
Sigma_X = cov(X, 1); % MLE for Sigma_X (normalized by N, not N-1)

% Step size function
epsilon_n = @(n) 1/(n+1);

% Store parameter estimates for plotting
beta_history = zeros(p, n_samples);
sigma2_history = zeros(1, n_samples);
mu_X_history = zeros(p, n_samples);
Sigma_X_history = zeros(p, p, n_samples);

%% Step 2: Sequential updating (Martingale Posterior)
for n = 1:n_samples
    xn = X(n, :)'; % Current predictor sample (column vector)
    yn = y(n); % Current response sample
    eps_n = epsilon_n(n); % Adaptive step size
    
    % Update beta
    beta = beta + eps_n * (xn * (yn - xn' * beta));
    
    % Update sigma^2
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
end

%% Display final estimates
fprintf('Estimated beta:\n');
disp(beta);
fprintf('Estimated sigma^2: %f\n', sigma2);
fprintf('Estimated mu_X:\n');
disp(mu_X);
fprintf('Estimated Sigma_X:\n');
disp(Sigma_X);

%% Step 3: Visualization

% Set figure properties for better visualization
set(0, 'DefaultFigureWindowStyle', 'normal'); % Normal window style for better plotting
set(0, 'DefaultAxesFontSize', 12); % Default axis font size
set(0, 'DefaultLineLineWidth', 1.5); % Default line width
set(0, 'DefaultAxesGridLineStyle', '-'); % Enable grid lines for better readability

% Plot estimation history of beta
figure('Position', [100, 100, 800, 900]); % Custom figure size
for i = 1:p
    subplot(p,1,i);
    plot(1:n_samples, beta_history(i, :), 'b-', 'LineWidth', 1.5); hold on;
    yline(beta_true(i), 'r--', 'LineWidth', 1.5); % True value in red dashed line
    xlabel('Iterations', 'FontSize', 12);
    ylabel(['\beta_' num2str(i)], 'FontSize', 12);
    title(['Estimation of \beta_' num2str(i)], 'FontSize', 14);
    legend('Estimated', 'True', 'Location', 'Best');
    grid on;
end

% Plot estimation history of sigma^2
figure('Position', [100, 100, 800, 600]);
plot(1:n_samples, sigma2_history, 'b-', 'LineWidth', 1.5); hold on;
yline(sigma2_true, 'r--', 'LineWidth', 1.5);
xlabel('Iterations', 'FontSize', 12);
ylabel('\sigma^2', 'FontSize', 12);
title('Estimation of \sigma^2', 'FontSize', 14);
legend('Estimated', 'True', 'Location', 'Best');
grid on;

% Plot estimation history of mu_X
figure('Position', [100, 100, 800, 900]); % Custom figure size
for i = 1:p
    subplot(p,1,i);
    plot(1:n_samples, mu_X_history(i, :), 'b-', 'LineWidth', 1.5); hold on;
    yline(mu_X_true(i), 'r--', 'LineWidth', 1.5);
    xlabel('Iterations', 'FontSize', 12);
    ylabel(['\mu_X_' num2str(i)], 'FontSize', 12);
    title(['Estimation of \mu_X_' num2str(i)], 'FontSize', 14);
    legend('Estimated', 'True', 'Location', 'Best');
    grid on;
end

% Plot estimation history of elements of Sigma_X
figure('Position', [100, 100, 1000, 1200]); % Custom figure size
idx = 1;
for i = 1:p
    for j = i:p  % Only upper triangular part including diagonal
        subplot(p, p, idx);
        plot(1:n_samples, squeeze(Sigma_X_history(i, j, :)), 'b-', 'LineWidth', 1.5); hold on;
        yline(Sigma_X_true(i, j), 'r--', 'LineWidth', 1.5);
        xlabel('Iterations', 'FontSize', 12);
        ylabel(['\Sigma_{' num2str(i) ',' num2str(j) '}'], 'FontSize', 12);
        title(['Estimation of \Sigma_{' num2str(i) ',' num2str(j) '}'], 'FontSize', 14);
        legend('Estimated', 'True', 'Location', 'Best');
        grid on;
        idx = idx + 1;
    end
end
