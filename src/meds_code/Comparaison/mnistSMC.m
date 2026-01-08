clc; clear; close all;
tic;

%% 1. Load MNIST and Preprocess
[XTrain, YTrain] = digitTrain4DArrayData;
XTrain = reshape(XTrain, [28*28, size(XTrain,4)])';
YTrain = grp2idx(YTrain) - 1;

n_samples = 1000;
X = XTrain(1:n_samples, :);
Y = YTrain(1:n_samples);

%% 2. Network Structure
input_size = 784;
hidden_size = 32;
output_size = 10;

%% 3. SMC Parameters
N_particles = 50;
theta_particles = zeros(N_particles, (input_size+1)*hidden_size + (hidden_size+1)*output_size);
weights = ones(N_particles,1) / N_particles;

for i = 1:N_particles
    theta_particles(i,:) = initialize_theta(input_size, hidden_size, output_size);
end

entropy_loss = zeros(n_samples,1);
mse_loss = zeros(n_samples,1);
accuracy_trace = zeros(n_samples,1);
time_per_iter = zeros(n_samples,1);

%% 4. SMC Inference Loop
for t = 1:n_samples
    tic_iter = tic;
    x = X(t,:)';
    y = Y(t);
    log_likelihoods = zeros(N_particles,1);

    for p = 1:N_particles
        theta = theta_particles(p,:)';
        [probs, ~] = forward_pass(x, theta, input_size, hidden_size, output_size);
        log_likelihoods(p) = log(probs(y+1) + 1e-8);
    end

    weights = weights .* exp(log_likelihoods - max(log_likelihoods));
    weights = weights / sum(weights + 1e-12);

    % Resample if ESS low
    Neff = 1/sum(weights.^2);
    if Neff < 0.5 * N_particles
        indices = randsample(1:N_particles, N_particles, true, weights);
        theta_particles = theta_particles(indices,:);
        weights = ones(N_particles,1) / N_particles;
    end

    % Compute metrics
    [acc, mse, ce] = ensemble_metrics(theta_particles, X(1:t,:), Y(1:t), input_size, hidden_size, output_size);
    accuracy_trace(t) = acc;
    mse_loss(t) = mse;
    entropy_loss(t) = ce;
    time_per_iter(t) = toc(tic_iter);
end

%% 5. Posterior Stats
posterior_mean = mean(theta_particles,1);
posterior_std = std(theta_particles,0,1);

fprintf('\nFinal Accuracy: %.4f\n', accuracy_trace(end));
fprintf('Final MSE: %.4f\n', mse_loss(end));
fprintf('Final Cross-Entropy: %.4f\n', entropy_loss(end));
fprintf('Mean Posterior Std: %.6f\n', mean(posterior_std));
fprintf('Total Time: %.2f sec\n', toc);

%% 6. Visualization

% Accuracy
figure;
plot(1:n_samples, accuracy_trace, 'LineWidth', 2);
xlabel('Sample'); ylabel('Accuracy');
title('Ensemble Accuracy Over Time');

% Losses
figure;
plot(mse_loss, 'r-', 'LineWidth', 1.5); hold on;
plot(entropy_loss, 'b-', 'LineWidth', 1.5);
xlabel('Sample'); ylabel('Loss');
legend('MSE','Cross-Entropy');
title('Loss Metrics Over Time');

% Posterior uncertainty
figure;
bar(posterior_std);
xlabel('Parameter Index');
ylabel('Posterior Std Dev');
title('Uncertainty in Parameters (Posterior Std)');

% Mean ± Std visualization
figure;
plot(posterior_mean, 'k'); hold on;
plot(posterior_mean + posterior_std, 'g--');
plot(posterior_mean - posterior_std, 'g--');
xlabel('Parameter Index'); ylabel('Value');
title('Posterior Mean ± Std');

% Time per iteration
figure;
plot(time_per_iter, 'LineWidth', 1.5);
xlabel('Sample'); ylabel('Time (s)');
title('Computational Time per SMC Iteration');

%% ====================
% Helper Functions Below
% ====================

function theta = initialize_theta(n_in, n_hid, n_out)
    W1 = randn(n_hid, n_in)*0.05;
    b1 = randn(n_hid,1)*0.01;
    W2 = randn(n_out, n_hid)*0.05;
    b2 = randn(n_out,1)*0.01;
    theta = [W1(:); b1; W2(:); b2];
end

function [probs, a1] = forward_pass(x, theta, n_in, n_hid, n_out)
    [W1, b1, W2, b2] = unpack_theta(theta, n_in, n_hid, n_out);
    z1 = W1*x + b1;
    a1 = max(0, z1);  % ReLU
    z2 = W2*a1 + b2;
    exps = exp(z2 - max(z2));
    probs = exps / sum(exps);
end

function [W1, b1, W2, b2] = unpack_theta(theta, n_in, n_hid, n_out)
    idx1 = n_hid*n_in;
    W1 = reshape(theta(1:idx1), n_hid, n_in);
    idx2 = idx1 + n_hid;
    b1 = theta(idx1+1:idx2);
    idx3 = idx2 + n_out*n_hid;
    W2 = reshape(theta(idx2+1:idx3), n_out, n_hid);
    b2 = theta(idx3+1:end);
end

function [acc, mse_total, ce_total] = ensemble_metrics(theta_particles, X, Y, n_in, n_hid, n_out)
    N = size(X,1);
    num_particles = size(theta_particles,1);
    probs_all = zeros(N, n_out);
    
    for p = 1:num_particles
        theta = theta_particles(p,:)';
        [W1, b1, W2, b2] = unpack_theta(theta, n_in, n_hid, n_out);
        Z1 = max(0, X * W1' + b1');
        Z2 = Z1 * W2' + b2';
        exps = exp(Z2 - max(Z2,[],2));
        probs = exps ./ sum(exps,2);
        probs_all = probs_all + probs;
    end

    probs_all = probs_all / num_particles;
    [~, final_preds] = max(probs_all, [], 2);
    final_preds = final_preds - 1;
    acc = mean(final_preds == Y);
    
    Y_onehot = zeros(N, n_out);
    for i = 1:N
        Y_onehot(i, Y(i)+1) = 1;
    end
    mse_total = mean((Y_onehot - probs_all).^2, 'all');

    log_probs = log(probs_all + 1e-8);
    ce_total = -mean(sum(Y_onehot .* log_probs,2));
end
