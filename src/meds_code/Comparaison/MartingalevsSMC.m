clc; clear; close all;
tic;

%% 1. Load and Preprocess MNIST Data
[XTrain, YTrain] = digitTrain4DArrayData;
XTrain = reshape(XTrain, [28*28, size(XTrain,4)])';
YTrain = grp2idx(YTrain) - 1; % Convert labels to numeric

% Use a subset for demonstration
n_samples = 1000;
X = XTrain(1:n_samples, :);
Y = YTrain(1:n_samples);

%% 2. Define Neural Network Architecture
input_size = 784;
hidden_size = 32;
output_size = 10;

%% 3. Initialize Parameters for SMC and Martingale Posterior
N_particles = 50;
theta_dim = (input_size + 1)*hidden_size + (hidden_size + 1)*output_size;

% Initialize particles for SMC
theta_particles_smc = zeros(N_particles, theta_dim);
weights_smc = ones(N_particles,1) / N_particles;

% Initialize particles for Martingale Posterior
theta_particles_mp = zeros(N_particles, theta_dim);
weights_mp = ones(N_particles,1) / N_particles;

for i = 1:N_particles
    theta_particles_smc(i,:) = initialize_theta(input_size, hidden_size, output_size);
    theta_particles_mp(i,:) = initialize_theta(input_size, hidden_size, output_size);
end

% Initialize metrics storage
metrics_smc = struct('accuracy', zeros(n_samples,1), 'mse', zeros(n_samples,1), 'cross_entropy', zeros(n_samples,1));
metrics_mp = struct('accuracy', zeros(n_samples,1), 'mse', zeros(n_samples,1), 'cross_entropy', zeros(n_samples,1));

%% 4. Training Loop
for t = 1:n_samples
    x = X(t,:)';
    y = Y(t);
    
    % One-hot encoding for the label
    y_onehot = zeros(output_size,1); y_onehot(y+1) = 1;
    
    %% SMC Update
    log_likelihoods_smc = zeros(N_particles,1);
    for p = 1:N_particles
        theta = theta_particles_smc(p,:)';
        [probs, ~] = forward_pass(x, theta, input_size, hidden_size, output_size);
        log_likelihoods_smc(p) = log(probs(y+1) + 1e-8);
    end
    weights_smc = weights_smc .* exp(log_likelihoods_smc - max(log_likelihoods_smc));
    weights_smc = weights_smc / sum(weights_smc + 1e-12);
    
    % Resample if effective sample size is low
    Neff_smc = 1 / sum(weights_smc.^2);
    if Neff_smc < 0.5 * N_particles
        indices = randsample(1:N_particles, N_particles, true, weights_smc);
        theta_particles_smc = theta_particles_smc(indices,:);
        weights_smc = ones(N_particles,1) / N_particles;
    end
    
    % Compute ensemble metrics for SMC
    [acc_smc, mse_smc, ce_smc] = ensemble_metrics(theta_particles_smc, X(1:t,:), Y(1:t), input_size, hidden_size, output_size);
    metrics_smc.accuracy(t) = acc_smc;
    metrics_smc.mse(t) = mse_smc;
    metrics_smc.cross_entropy(t) = ce_smc;
    
    %% Martingale Posterior Update
    log_likelihoods_mp = zeros(N_particles,1);
    for p = 1:N_particles
        theta = theta_particles_mp(p,:)';
        [probs, ~] = forward_pass(x, theta, input_size, hidden_size, output_size);
        log_likelihoods_mp(p) = log(probs(y+1) + 1e-8);
    end
    weights_mp = weights_mp .* exp(log_likelihoods_mp - max(log_likelihoods_mp));
    weights_mp = weights_mp / sum(weights_mp + 1e-12);
    
    % Resample if effective sample size is low
    Neff_mp = 1 / sum(weights_mp.^2);
    if Neff_mp < 0.5 * N_particles
        indices = randsample(1:N_particles, N_particles, true, weights_mp);
        theta_particles_mp = theta_particles_mp(indices,:);
        weights_mp = ones(N_particles,1) / N_particles;
    end
    
    % Compute ensemble metrics for Martingale Posterior
    [acc_mp, mse_mp, ce_mp] = ensemble_metrics(theta_particles_mp, X(1:t,:), Y(1:t), input_size, hidden_size, output_size);
    metrics_mp.accuracy(t) = acc_mp;
    metrics_mp.mse(t) = mse_mp;
    metrics_mp.cross_entropy(t) = ce_mp;
end

%% 5. Posterior Analysis
posterior_mean_smc = mean(theta_particles_smc,1);
posterior_std_smc = std(theta_particles_smc,0,1);

posterior_mean_mp = mean(theta_particles_mp,1);
posterior_std_mp = std(theta_particles_mp,0,1);

fprintf('\nSMC Final Accuracy: %.4f\n', metrics_smc.accuracy(end));
fprintf('SMC Final MSE: %.4f\n', metrics_smc.mse(end));
fprintf('SMC Final Cross-Entropy: %.4f\n', metrics_smc.cross_entropy(end));
fprintf('SMC Mean Posterior Std: %.6f\n', mean(posterior_std_smc));

fprintf('\nMartingale Posterior Final Accuracy: %.4f\n', metrics_mp.accuracy(end));
fprintf('Martingale Posterior Final MSE: %.4f\n', metrics_mp.mse(end));
fprintf('Martingale Posterior Final Cross-Entropy: %.4f\n', metrics_mp.cross_entropy(end));
fprintf('Martingale Posterior Mean Posterior Std: %.6f\n', mean(posterior_std_mp));

toc;

%% 6. Visualization

% Accuracy
figure;
plot(1:n_samples, metrics_smc.accuracy, 'b-', 'LineWidth', 2); hold on;
plot(1:n_samples, metrics_mp.accuracy, 'r--', 'LineWidth', 2);
xlabel('Sample Index'); ylabel('Accuracy');
legend('SMC','Martingale Posterior'); grid on;
title('Classification Accuracy over Time');

% MSE
figure;
plot(1:n_samples, metrics_smc.mse, 'b-', 'LineWidth', 2); hold on;
plot(1:n_samples, metrics_mp.mse, 'r--', 'LineWidth', 2);
xlabel('Sample Index'); ylabel('MSE');
legend('SMC','Martingale Posterior'); grid on;
title('Mean Squared Error (MSE) over Time');

% Cross Entropy
figure;
plot(1:n_samples, metrics_smc.cross_entropy, 'b-', 'LineWidth', 2); hold on;
plot(1:n_samples, metrics_mp.cross_entropy, 'r--', 'LineWidth', 2);
xlabel('Sample Index'); ylabel('Cross-Entropy');
legend('SMC','Martingale Posterior'); grid on;
title('Cross-Entropy Loss over Time');

% Posterior std (Uncertainty)
figure;
subplot(2,1,1);
bar(posterior_std_smc, 'FaceColor', 'b'); title('SMC: Posterior Std Dev'); ylabel('\sigma'); xlabel('Parameter Index');

subplot(2,1,2);
bar(posterior_std_mp, 'FaceColor', 'r'); title('Martingale Posterior: Posterior Std Dev'); ylabel('\sigma'); xlabel('Parameter Index');

% Histogram of Parameter Distributions
figure;
subplot(1,2,1);
histogram(posterior_std_smc, 20, 'FaceColor', 'b'); title('SMC Posterior Std Distribution'); xlabel('\sigma'); ylabel('Count');

subplot(1,2,2);
histogram(posterior_std_mp, 20, 'FaceColor', 'r'); title('Martingale Posterior Std Distribution'); xlabel('\sigma'); ylabel('Count');

% Computational Time Comparison
fprintf('\n--- Computational Time ---\n');
disp(['Total Time: ', num2str(toc, '%.2f'), ' seconds']);

%% Helper Functions

function theta = initialize_theta(input_size, hidden_size, output_size)
    W1 = randn(hidden_size, input_size) * sqrt(2 / (input_size + hidden_size));
    b1 = zeros(hidden_size, 1);
    W2 = randn(output_size, hidden_size) * sqrt(2 / (hidden_size + output_size));
    b2 = zeros(output_size, 1);
    theta = [W1(:); b1; W2(:); b2];
end

function [probs, activations] = forward_pass(x, theta, input_size, hidden_size, output_size)
    W1 = reshape(theta(1:hidden_size*input_size), hidden_size, input_size);
    idx = hidden_size*input_size;
    b1 = theta(idx+1 : idx+hidden_size);
    idx = idx + hidden_size;
    W2 = reshape(theta(idx+1 : idx+output_size*hidden_size), output_size, hidden_size);
    b2 = theta(end-output_size+1:end);

    z1 = W1 * x + b1;
    a1 = tanh(z1);
    z2 = W2 * a1 + b2;
    exp_z = exp(z2 - max(z2));
    probs = exp_z / sum(exp_z);

    activations = struct('z1', z1, 'a1', a1, 'z2', z2);
end

function [acc, mse, cross_ent] = ensemble_metrics(theta_particles, X, Y, input_size, hidden_size, output_size)
    N_particles = size(theta_particles,1);
    n = size(X,1);
    probs_ensemble = zeros(output_size, n);

    for p = 1:N_particles
        for i = 1:n
            x = X(i,:)';
            theta = theta_particles(p,:)';
            [probs, ~] = forward_pass(x, theta, input_size, hidden_size, output_size);
            probs_ensemble(:,i) = probs_ensemble(:,i) + probs;
        end
    end

    probs_ensemble = probs_ensemble / N_particles;
    [~, preds] = max(probs_ensemble, [], 1);
    acc = mean(preds' - 1 == Y);

    Y_onehot = full(ind2vec(Y'+1, output_size));
    mse = mean((probs_ensemble - Y_onehot).^2, 'all');
    cross_ent = -mean(log(probs_ensemble(Y'+1 + (0:n-1)*output_size) + 1e-8));
end
