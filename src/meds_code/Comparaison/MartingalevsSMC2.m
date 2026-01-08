clc;
clear;
close all;
tic;

%% 1. Load MNIST and Preprocess
[XTrain, YTrain] = digitTrain4DArrayData;
XTrain = reshape(XTrain, [28*28, size(XTrain,4)])';  % Flatten
YTrain = grp2idx(YTrain) - 1;  % Numeric labels from 0

% Use a subset for quick demo
n_samples = 1000;
X = XTrain(1:n_samples, :);
Y = YTrain(1:n_samples);

%% 2. Network Architecture
input_size = 784;
hidden_size = 32;
output_size = 10;

%% 3. Martingale Posterior Sampling (MPS)
D = 10; % Number of parallel samples (Martingales)
theta_mps = zeros(D, (input_size + 1) * hidden_size + (hidden_size + 1) * output_size);

% Initialize MPS Parameters
for i = 1:D
    theta_mps(i,:) = initialize_theta(input_size, hidden_size, output_size);
end

epochs = 10;  % Number of epochs
batch_size = 100;  % Size of each mini-batch
accuracy_mps = zeros(epochs, 1);
mse_mps = zeros(epochs, 1);
entropy_mps = zeros(epochs, 1);

for epoch = 1:epochs
    for t = 1:n_samples
        x = X(t, :)';
        y = Y(t);
        y_onehot = zeros(output_size, 1);
        y_onehot(y + 1) = 1;

        for d = 1:D
            theta = theta_mps(d,:)';
            [probs, ~] = forward_pass(x, theta, input_size, hidden_size, output_size);
            log_likelihood = log(probs(y + 1) + 1e-8);  % avoid log(0)
            theta_mps(d,:) = theta_mps(d,:) + (1/(t+1)) * log_likelihood;
        end
        
        % Evaluation metrics for MPS
        [accuracy_mps(epoch), mse_mps(epoch), entropy_mps(epoch)] = ensemble_metrics(theta_mps, X(1:t,:), Y(1:t,:), input_size, hidden_size, output_size);
    end
end

%% 4. Sequential Monte Carlo (SMC)
N_particles = 50;
theta_smc = zeros(N_particles, (input_size + 1) * hidden_size + (hidden_size + 1) * output_size);
weights_smc = ones(N_particles, 1) / N_particles;
log_likelihoods_smc = zeros(N_particles, 1);

for i = 1:N_particles
    theta_smc(i,:) = initialize_theta(input_size, hidden_size, output_size);
end

accuracy_smc = zeros(epochs, 1);
mse_smc = zeros(epochs, 1);
entropy_smc = zeros(epochs, 1);

for epoch = 1:epochs
    for t = 1:n_samples
        x = X(t,:)';
        y = Y(t);
        y_onehot = zeros(output_size, 1);
        y_onehot(y + 1) = 1;
        
        for p = 1:N_particles
            theta = theta_smc(p,:)';
            [probs, ~] = forward_pass(x, theta, input_size, hidden_size, output_size);
            log_likelihoods_smc(p) = log(probs(y + 1) + 1e-8);  % avoid log(0)
        end
        
        % Update weights using likelihood
        weights_smc = weights_smc .* exp(log_likelihoods_smc - max(log_likelihoods_smc));
        weights_smc = weights_smc / sum(weights_smc + 1e-12);  % normalize

        % Resampling if effective sample size is low
        Neff = 1 / sum(weights_smc.^2);
        if Neff < 0.5 * N_particles
            indices = randsample(1:N_particles, N_particles, true, weights_smc);
            theta_smc = theta_smc(indices,:);
            weights_smc = ones(N_particles, 1) / N_particles;
        end

        % Evaluation metrics for SMC
        [accuracy_smc(epoch), mse_smc(epoch), entropy_smc(epoch)] = ensemble_metrics(theta_smc, X(1:t,:), Y(1:t,:), input_size, hidden_size, output_size);
    end
end

%% 5. Posterior Analysis for MPS and SMC
posterior_mean_mps = mean(theta_mps, 1);
posterior_std_mps = std(theta_mps, 0, 1);

posterior_mean_smc = mean(theta_smc, 1);
posterior_std_smc = std(theta_smc, 0, 1);

fprintf('MPS Final Accuracy: %.4f\n', accuracy_mps(end));
fprintf('SMC Final Accuracy: %.4f\n', accuracy_smc(end));

%% 6. Plots

% 6.1: Accuracy over epochs
figure;
subplot(2,2,1);
plot(accuracy_mps, 'LineWidth', 2, 'DisplayName', 'MPS');
hold on;
plot(accuracy_smc, 'LineWidth', 2, 'DisplayName', 'SMC');
xlabel('Epochs'); ylabel('Accuracy');
title('Accuracy over Epochs');
legend;

% 6.2: MSE and Cross-Entropy over epochs
subplot(2,2,2);
plot(mse_mps, 'r', 'LineWidth', 1.5); hold on;
plot(mse_smc, 'g', 'LineWidth', 1.5);
xlabel('Epochs'); legend('MPS MSE','SMC MSE'); title('MSE Loss over Epochs');

subplot(2,2,3);
plot(entropy_mps, 'b', 'LineWidth', 1.5); hold on;
plot(entropy_smc, 'k', 'LineWidth', 1.5);
xlabel('Epochs'); legend('MPS Entropy','SMC Entropy'); title('Cross-Entropy over Epochs');

% 6.3: Posterior Std of Weights and Biases (Uncertainty Quantification)
subplot(2,2,4);
imagesc(reshape(posterior_std_mps, [hidden_size, input_size + 1])); colorbar;
title('MPS Posterior Std for Weights (Layer 1)');
xlabel('Input/Hidden Nodes'); ylabel('Hidden Units');

figure;
imagesc(reshape(posterior_std_smc, [hidden_size, input_size + 1])); colorbar;
title('SMC Posterior Std for Weights (Layer 1)');
xlabel('Input/Hidden Nodes'); ylabel('Hidden Units');

%% Helper Functions

function theta = initialize_theta(n_in, n_hid, n_out)
    W1 = randn(n_hid, n_in) * 0.05;
    b1 = randn(n_hid, 1) * 0.01;
    W2 = randn(n_out, n_hid) * 0.05;
    b2 = randn(n_out, 1) * 0.01;
    theta = [W1(:); b1; W2(:); b2];
end

function [probs, a1] = forward_pass(x, theta, n_in, n_hid, n_out)
    [W1, b1, W2, b2] = unpack_theta(theta, n_in, n_hid, n_out);
    z1 = W1 * x + b1;
    a1 = max(0, z1);  % ReLU
    z2 = W2 * a1 + b2;
    exps = exp(z2 - max(z2));
    probs = exps / sum(exps);
end

function [W1, b1, W2, b2] = unpack_theta(theta, n_in, n_hid, n_out)
    idx1 = n_hid * n_in;
    W1 = reshape(theta(1:idx1), n_hid, n_in);
    idx2 = idx1 + n_hid;
    b1 = theta(idx1 + 1:idx2);
    idx3 = idx2 + n_out * n_hid;
    W2 = reshape(theta(idx2 + 1:idx3), n_out, n_hid);
    b2 = theta(idx3 + 1:end);
end

function [acc, mse_total, ce_total] = ensemble_metrics(theta_particles, X, Y, n_in, n_hid, n_out)
    N = size(X,1);
    num_particles = size(theta_particles,1);
    preds_all = zeros(N, num_particles);
    probs_all = zeros(N, n_out);
    
    for p = 1:num_particles
        theta = theta_particles(p,:)';
        [W1, b1, W2, b2] = unpack_theta(theta, n_in, n_hid, n_out);
        Z1 = max(0, X * W1' + b1');
        Z2 = Z1 * W2' + b2';
        exps = exp(Z2 - max(Z2,[],2));
        probs = exps ./ sum(exps,2);
        [~, preds] = max(probs, [], 2);
        preds = preds - 1;
        preds_all(:,p) = preds;
        probs_all = probs_all + probs;
    end

    probs_all = probs_all / num_particles;
    [~, final_preds] = max(probs_all, [], 2);
    final_preds = final_preds - 1;
    acc = mean(final_preds == Y);
    
    % MSE
    Y_onehot = zeros(N, n_out);
    for i = 1:N
        Y_onehot(i, Y(i) + 1) = 1;
    end
    mse_total = mean((Y_onehot - probs_all).^2, 'all');

    % Cross-Entropy
    log_probs = log(probs_all + 1e-8);
    ce_total = -mean(sum(Y_onehot .* log_probs, 2));
end
