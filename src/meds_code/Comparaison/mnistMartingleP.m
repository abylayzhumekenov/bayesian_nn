% Small-scale DNN with Martingale Posterior Sampling (Parallelized with Plots)

clear; clc; close all;
tic;

% Load MNIST (Digit Dataset from MATLAB)
[XTrain, YTrain] = digitTrain4DArrayData;
XTrain = reshape(XTrain, [28*28, size(XTrain,4)])';  % Flatten images
YTrain = grp2idx(YTrain) - 1;  % Convert labels to 0-9

% Network Parameters
input_size = 784;
hidden_size = 32;
output_size = 10;
num_samples = 5000;  % Small scale for faster computation
epochs = 100;
D = 10;  % Number of parallel martingales

% Subsample Data
X = XTrain(1:num_samples, :);
Y = YTrain(1:num_samples);

% Initialize network
theta_init = initialize_theta(input_size, hidden_size, output_size);

% Preallocate
theta_samples = zeros(D, numel(theta_init));

% Parallel Martingale Posterior Sampling
parfor d = 1:D
    theta_d = theta_init;
    for m = 1:num_samples
        x_m = X(m, :)';
        y_m = Y(m);
        s = score_function(x_m, y_m, theta_d, input_size, hidden_size, output_size);
        theta_d = theta_d + (1/(m+1)) * s;  % Martingale update
    end
    theta_samples(d, :) = theta_d;
end

% Posterior Statistics
theta_mean = mean(theta_samples, 1);
theta_std = std(theta_samples, [], 1);

% Predictions
[preds, probs] = predict(X, theta_mean, input_size, hidden_size, output_size);

% Evaluation Metrics
accuracy = mean(preds == Y);
log_likelihood = mean(log(max(probs(sub2ind(size(probs), (1:num_samples)', Y+1)), 1e-10)));
mse = mean((preds - Y).^2);
entropy_loss = -mean(sum(probs .* log(max(probs, 1e-10)), 2));
cost = toc;

% Display
fprintf('\n==== Evaluation ====\n');
fprintf('Accuracy        : %.4f\n', accuracy);
fprintf('Log-Likelihood  : %.4f\n', log_likelihood);
fprintf('MSE             : %.4f\n', mse);
fprintf('Entropy Loss    : %.4f\n', entropy_loss);
fprintf('Posterior Std µ : %.6f\n', mean(theta_std));
fprintf('Computational Time (s): %.2f\n', cost);

%% ===================== PLOTS ==========================

% 1. Accuracy Bar
figure;
bar(accuracy * 100);
ylim([0 100]);
title('Classification Accuracy');
ylabel('Accuracy (%)');
xticklabels({'Martingale Posterior'});
grid on;

% 2. Posterior Std Distribution
figure;
histogram(theta_std, 30);
title('Posterior Std Distribution');
xlabel('Standard Deviation of Parameters');
ylabel('Frequency');
grid on;

% 3. Posterior Mean ± Std for First 50 Parameters
figure;
sample_idx = 1:50;
errorbar(sample_idx, theta_mean(sample_idx), theta_std(sample_idx), 'o');
title('Posterior Mean ± Std (First 50 Params)');
xlabel('Parameter Index');
ylabel('Value');
grid on;

% 4. Computational Cost
figure;
bar(cost);
title('Computational Cost');
ylabel('Time (seconds)');
xticklabels({'Martingale Sampling'});
grid on;

%% ===================== FUNCTIONS ==========================

function theta = initialize_theta(n_in, n_hid, n_out)
    W1 = randn(n_hid, n_in) * 0.01;
    b1 = zeros(n_hid, 1);
    W2 = randn(n_out, n_hid) * 0.01;
    b2 = zeros(n_out, 1);
    theta = [W1(:); b1(:); W2(:); b2(:)];
end

function s = score_function(x, y, theta, n_in, n_hid, n_out)
    [W1, b1, W2, b2] = unpack_theta(theta, n_in, n_hid, n_out);
    z1 = W1 * x + b1;
    a1 = max(0, z1);
    z2 = W2 * a1 + b2;
    exp_scores = exp(z2 - max(z2));
    probs = exp_scores / sum(exp_scores);
    y_onehot = zeros(n_out,1); y_onehot(y+1) = 1;
    dz2 = probs - y_onehot;
    dW2 = dz2 * a1';
    db2 = dz2;
    da1 = W2' * dz2;
    dz1 = da1 .* (z1 > 0);
    dW1 = dz1 * x';
    db1 = dz1;
    grad = [dW1(:); db1(:); dW2(:); db2(:)];
    s = grad;
end

function [W1, b1, W2, b2] = unpack_theta(theta, n_in, n_hid, n_out)
    idx1 = n_hid * n_in;
    W1 = reshape(theta(1:idx1), n_hid, n_in);
    idx2 = idx1 + n_hid;
    b1 = reshape(theta(idx1+1:idx2), n_hid, 1);
    idx3 = idx2 + n_out * n_hid;
    W2 = reshape(theta(idx2+1:idx3), n_out, n_hid);
    b2 = reshape(theta(idx3+1:end), n_out, 1);
end

function [preds, probs] = predict(X, theta, n_in, n_hid, n_out)
    [W1, b1, W2, b2] = unpack_theta(theta, n_in, n_hid, n_out);
    Z1 = max(0, X * W1' + b1');
    Z2 = Z1 * W2' + b2';
    exp_scores = exp(Z2 - max(Z2, [], 2));
    probs = exp_scores ./ sum(exp_scores, 2);
    [~, preds] = max(probs, [], 2);
    preds = preds - 1;
end
