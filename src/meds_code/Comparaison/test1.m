% Martingale Posterior vs SMC on MNIST - Full Script with Visualizations
clc; clear; close all;
tic;

%% 1. Load and Preprocess MNIST Data
[XTrain, YTrain] = digitTrain4DArrayData;
XTrain = reshape(XTrain, [28*28, size(XTrain,4)])';
YTrain = grp2idx(YTrain) - 1; % Convert labels to 0-based

n_samples = 1000;
X = XTrain(1:n_samples, :);
Y = YTrain(1:n_samples);

%% 2. Neural Network Parameters
input_size = 784;
hidden_size = 32;
output_size = 10;

%% 3. Initialization for MLE
theta_mle = train_mle(X, Y, input_size, hidden_size, output_size);

%% 4. Martingale Posterior Sampling Parameters
N_particles = 50;
T = 200; % Steps per trajectory

% Score function step size
epsilon = @(m) 1 / (m + 1);

% Initialize particles for Martingale
theta_mp_particles = zeros(N_particles, numel(theta_mle));
parfor d = 1:N_particles
    theta = theta_mle;
    for m = 1:T
        idx = randi(n_samples);
        x = X(idx,:)';
        y = Y(idx);
        s = score_function(x, y, theta, input_size, hidden_size, output_size);
        theta = theta + epsilon(m) * s;
    end
    theta_mp_particles(d,:) = theta;
end

%% 5. SMC Initialization
theta_smc_particles = zeros(N_particles, numel(theta_mle));
weights_smc = ones(N_particles,1) / N_particles;
for i = 1:N_particles
    theta_smc_particles(i,:) = theta_mle + 0.01 * randn(size(theta_mle));
end

%% 6. Evaluation Metrics
metrics = struct('accuracy', [], 'mse', [], 'cross_entropy', [], 'log_likelihood', [], 'cost', []);
[metrics.mp] = evaluate_ensemble(theta_mp_particles, X, Y, input_size, hidden_size, output_size);
[metrics.smc] = evaluate_ensemble(theta_smc_particles, X, Y, input_size, hidden_size, output_size);

%% 7. Visualization
figure;
tiledlayout(2,3);

nexttile; bar([metrics.mp.accuracy, metrics.smc.accuracy]); title('Accuracy'); legend('MP','SMC');
nexttile; bar([metrics.mp.mse, metrics.smc.mse]); title('MSE');
nexttile; bar([metrics.mp.cross_entropy, metrics.smc.cross_entropy]); title('Cross-Entropy');
nexttile; bar([metrics.mp.log_likelihood, metrics.smc.log_likelihood]); title('Log-Likelihood');
nexttile; bar([metrics.mp.cost, metrics.smc.cost]); title('Cost (s)');
nexttile; bar([std(theta_mp_particles), std(theta_smc_particles)]'); title('Parameter Std Dev'); legend('MP','SMC');

%% 8. Helper Functions
function theta = train_mle(X, Y, input_size, hidden_size, output_size)
    theta = initialize_theta(input_size, hidden_size, output_size);
    alpha = 0.05;
    for epoch = 1:5
        for i = 1:size(X,1)
            x = X(i,:)'; y = Y(i);
            grad = score_function(x, y, theta, input_size, hidden_size, output_size);
            theta = theta + alpha * grad;
        end
    end
end

function theta = initialize_theta(input_size, hidden_size, output_size)
    W1 = randn(hidden_size, input_size) * 0.1;
    b1 = zeros(hidden_size, 1);
    W2 = randn(output_size, hidden_size) * 0.1;
    b2 = zeros(output_size, 1);
    theta = [W1(:); b1; W2(:); b2];
end

function s = score_function(x, y, theta, input_size, hidden_size, output_size)
    [probs, act] = forward_pass(x, theta, input_size, hidden_size, output_size);
    y_onehot = zeros(output_size,1); y_onehot(y+1) = 1;
    dL = probs - y_onehot;
    W1 = reshape(theta(1:hidden_size*input_size), hidden_size, input_size);
    idx = hidden_size*input_size;
    b1 = theta(idx+1 : idx+hidden_size);
    idx = idx + hidden_size;
    W2 = reshape(theta(idx+1 : idx+output_size*hidden_size), output_size, hidden_size);
    b2 = theta(end-output_size+1:end);
    
    dz2 = dL;
    dW2 = dz2 * act.a1';
    db2 = dz2;
    da1 = W2' * dz2;
    dz1 = da1 .* (1 - act.a1.^2);
    dW1 = dz1 * x';
    db1 = dz1;
    s = [dW1(:); db1; dW2(:); db2];
end

function [probs, act] = forward_pass(x, theta, input_size, hidden_size, output_size)
    W1 = reshape(theta(1:hidden_size*input_size), hidden_size, input_size);
    idx = hidden_size*input_size;
    b1 = theta(idx+1 : idx+hidden_size);
    idx = idx + hidden_size;
    W2 = reshape(theta(idx+1 : idx+output_size*hidden_size), output_size, hidden_size);
    b2 = theta(end-output_size+1:end);
    z1 = W1 * x + b1;
    a1 = tanh(z1);
    z2 = W2 * a1 + b2;
    e = exp(z2 - max(z2));
    probs = e / sum(e);
    act = struct('a1', a1);
end

function metrics = evaluate_ensemble(particles, X, Y, input_size, hidden_size, output_size)
    N = size(particles,1);
    n = size(X,1);
    ensemble_probs = zeros(output_size, n);
    logL = 0;
    for i = 1:N
        for j = 1:n
            x = X(j,:)';
            [probs, ~] = forward_pass(x, particles(i,:)', input_size, hidden_size, output_size);
            ensemble_probs(:,j) = ensemble_probs(:,j) + probs;
            logL = logL + log(probs(Y(j)+1) + 1e-8);
        end
    end
    ensemble_probs = ensemble_probs / N;
    [~, preds] = max(ensemble_probs);
    acc = mean(preds' - 1 == Y);
    Y_onehot = full(ind2vec(Y'+1));
    mse = mean((ensemble_probs - Y_onehot).^2, 'all');
    ce = -mean(log(ensemble_probs(Y'+1 + (0:n-1)*output_size) + 1e-8));
    metrics.accuracy = acc;
    metrics.mse = mse;
    metrics.cross_entropy = ce;
    metrics.log_likelihood = logL / (N*n);
    metrics.cost = toc;
end