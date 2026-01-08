clear; clc; close all;
tStart = tic;

%% Load MNIST data (subset for speed)
[XTrain, YTrain] = digitTrain4DArrayData;
XTrain = reshape(XTrain, [28*28, size(XTrain,4)])';
YTrain = grp2idx(YTrain) - 1;

num_samples = 1000;
X = XTrain(1:num_samples, :);
Y = YTrain(1:num_samples);

%% Network architecture
input_size = 784;
hidden_size = 32;
output_size = 10;

%% Initial Parameters
theta_init = initialize_theta(input_size, hidden_size, output_size);

%% --- Martingale Posterior using SGLD ---
D = 100;  % number of posterior samples
batch_size = 20;
eta = 1e-3;
noise_scale = 1e-2;
theta_mart = zeros(D, numel(theta_init));

parfor d = 1:D
    theta_d = theta_init;
    for m = 1:num_samples
        i = randi(num_samples);
        x_i = X(i,:)'; y_i = Y(i);
        s = score_function(x_i, y_i, theta_d, input_size, hidden_size, output_size);
        s = s / (norm(s) + 1e-8); % normalize
        theta_d = theta_d + eta * s + noise_scale * randn(size(theta_d));
    end
    theta_mart(d,:) = theta_d;
end

%% --- Sequential Monte Carlo (SMC) ---
P = 20;
theta_smc = zeros(P, numel(theta_init));
for p = 1:P
    theta_smc(p,:) = theta_init + randn(size(theta_init)) * 0.01;
end

for m = 1:num_samples
    for p = 1:P
        x_m = X(m,:)'; y_m = Y(m);
        s = score_function(x_m, y_m, theta_smc(p,:)', input_size, hidden_size, output_size);
        theta_smc(p,:) = theta_smc(p,:) + 0.01 * s';
    end
end

%% --- MCMC (Metropolis-Hastings) ---
S = 1000;
theta_mcmc = zeros(S, numel(theta_init));
theta_mcmc(1,:) = theta_init';

for s = 2:S
    proposal = theta_mcmc(s-1,:) + randn(size(theta_init))' * 0.005;
    [~, prob1] = predict(X, theta_mcmc(s-1,:)', input_size, hidden_size, output_size);
    [~, prob2] = predict(X, proposal', input_size, hidden_size, output_size);
    logL1 = sum(log(prob1(sub2ind(size(prob1), (1:num_samples)', Y+1))));
    logL2 = sum(log(prob2(sub2ind(size(prob2), (1:num_samples)', Y+1))));
    if log(rand) < (logL2 - logL1)
        theta_mcmc(s,:) = proposal;
    else
        theta_mcmc(s,:) = theta_mcmc(s-1,:);
    end
end

%% --- Evaluation ---
mean_mart = mean(theta_mart);
mean_smc  = mean(theta_smc);
mean_mcmc = mean(theta_mcmc(end-100+1:end,:)); % Last 100

[preds_mart, ~] = predict(X, mean_mart', input_size, hidden_size, output_size);
[preds_smc, ~]  = predict(X, mean_smc',  input_size, hidden_size, output_size);
[preds_mcmc, ~] = predict(X, mean_mcmc', input_size, hidden_size, output_size);

acc_mart = mean(preds_mart == Y);
acc_smc  = mean(preds_smc == Y);
acc_mcmc = mean(preds_mcmc == Y);

mse_mart = mean((preds_mart - Y).^2);
mse_smc  = mean((preds_smc - Y).^2);
mse_mcmc = mean((preds_mcmc - Y).^2);

%% --- Visualization ---

% Posterior distributions
param_indices = 1:5;
figure;
for i = 1:length(param_indices)
    subplot(1,length(param_indices),i);
    histogram(theta_mart(:,param_indices(i)), 'Normalization','pdf', 'FaceAlpha',0.5); hold on;
    histogram(theta_smc(:,param_indices(i)), 'Normalization','pdf', 'FaceAlpha',0.5);
    histogram(theta_mcmc(end-100+1:end,param_indices(i)), 'Normalization','pdf', 'FaceAlpha',0.5);
    title(sprintf('\\theta_{%d}', param_indices(i)));
    if i == 1
        legend('Martingale-SGLD','SMC','MCMC');
    end
end
sgtitle('Posterior Distributions (selected parameters)');

% Accuracy & MSE
figure;
metrics = [acc_mart, acc_smc, acc_mcmc;
           mse_mart, mse_smc, mse_mcmc];
bar(metrics');
title('Accuracy and MSE');
legend('Accuracy','MSE');
xlabel('Method'); set(gca,'XTickLabel',{'Martingale','SMC','MCMC'});

% Posterior std
figure;
bar([std(theta_mart(:,1:10)); std(theta_smc(:,1:10)); std(theta_mcmc(end-100+1:end,1:10))]');
title('Posterior Std (first 10 parameters)');
legend('Martingale','SMC','MCMC');
xlabel('Parameter Index'); ylabel('Std');

fprintf('Total Runtime: %.2f seconds\n', toc(tStart));

%% --- Helper Functions ---
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
    dW2 = dz2 * a1'; db2 = dz2;
    da1 = W2' * dz2;
    dz1 = da1 .* (z1 > 0);
    dW1 = dz1 * x'; db1 = dz1;
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
    exp_scores = exp(Z2 - max(Z2,[],2));
    probs = exp_scores ./ sum(exp_scores,2);
    [~, preds] = max(probs, [], 2);
    preds = preds - 1;
end
