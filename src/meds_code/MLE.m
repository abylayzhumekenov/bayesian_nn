clear;
close all;
clc;
% Maximum Likelihood Estimation for Linear Regression

% Generate synthetic data
n = 100; % Number of data points
X = [ones(n,1), randn(n,1)]; % Design matrix with intercept
true_theta = [2; 3]; % True parameters
y = X * true_theta + randn(n,1); % Generate responses with noise

% Estimate theta using MLE (Ordinary Least Squares Solution)
theta_hat = (X' * X) \ (X' * y);

% Estimate sigma^2
sigma_hat_sq = (1/n) * sum((y - X * theta_hat).^2);

% Display results
fprintf('Estimated theta:\n');
disp(theta_hat);
fprintf('Estimated sigma^2: %.4f\n', sigma_hat_sq);

% High-Quality Plot
figure;
hold on;
grid on;
box on;
scatter(X(:,2), y, 50, 'b', 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.6);
plot(X(:,2), X * theta_hat, 'r', 'LineWidth', 2);

% Enhance Visualization
xlabel('Predictor Variable (X)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Response Variable (y)', 'FontSize', 12, 'FontWeight', 'bold');
title('Linear Regression with MLE', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Data Points', 'Fitted Regression Line'}, 'Location', 'best', 'FontSize', 12);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);

% Save figure
saveas(gcf, 'linear_regression_mle.png');
