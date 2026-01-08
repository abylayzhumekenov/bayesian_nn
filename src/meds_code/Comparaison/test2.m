% Martingale Posterior Sampling for DNN (1 Hidden Layer)
clear; clc; rng(1);

%% Simulated Data
N = 100; x = linspace(-3, 3, N)';
y = sin(x) + 0.3*randn(N,1);

%% Network Architecture
d_in = 1; d_hidden = 10; d_out = 1;
f = @(x, theta) theta.W2 * tanh(theta.W1 * x' + theta.b1) + theta.b2; % forward pass

%% Initialization
T = 200;        % number of updates
D = 50;         % number of posterior samples
theta0 = initParams(d_in, d_hidden, d_out); % initial guess
posterior_samples = cell(D,1);

%% Martingale Posterior Sampling
for d = 1:D
    theta = theta0;
    for t = 1:T
        idx = randi(N); x_t = x(idx); y_t = y(idx);
        [sW1, sb1, sW2, sb2] = scoreFunc(x_t, y_t, theta);
        eta = 1/(t + 1);
        % Update rule
        theta.W1 = theta.W1 + eta * sW1;
        theta.b1 = theta.b1 + eta * sb1;
        theta.W2 = theta.W2 + eta * sW2;
        theta.b2 = theta.b2 + eta * sb2;
    end
    posterior_samples{d} = theta;
end

%% Plot posterior samples of W1 elements
W1_samples = cell2mat(cellfun(@(th) th.W1(:)', posterior_samples, 'UniformOutput', false));
figure; 
for i = 1:min(5, size(W1_samples,2))
    subplot(2,3,i)
    histogram(W1_samples(:,i), 20, 'Normalization', 'pdf');
    title(['W1(', num2str(i), ') posterior']);
end

%% Plot posterior of basis (hidden unit outputs)
xx = linspace(-3, 3, 100);
Phi = zeros(D, length(xx), d_hidden);
for d = 1:D
    theta = posterior_samples{d};
    Phi(d,:,:) = tanh(theta.W1 * xx' + theta.b1);
end

figure;
for h = 1:min(5, d_hidden)
    subplot(2,3,h)
    plot(xx, squeeze(Phi(:,:,h))', 'Color', [0.5,0.5,1,0.3]);
    title(['Hidden Unit ', num2str(h), ' Basis']);
end

%% Functions
function theta = initParams(din, dh, dout)
    theta.W1 = 0.1*randn(dh, din);
    theta.b1 = 0.1*randn(dh,1);
    theta.W2 = 0.1*randn(dout, dh);
    theta.b2 = 0.1*randn(dout,1);
end

function [sW1, sb1, sW2, sb2] = scoreFunc(x, y, theta)
    % Forward pass
    a1 = theta.W1 * x + theta.b1;
    h = tanh(a1);
    y_hat = theta.W2 * h + theta.b2;
    % Gradients via chain rule (score is like gradient * residual)
    res = y - y_hat;
    dy = -res;
    dW2 = dy * h';
    db2 = dy;
    dh = theta.W2' * dy;
    da1 = dh .* (1 - tanh(a1).^2);  % derivative of tanh
    dW1 = da1 * x';
    db1 = da1;
    % Score: residual times gradient
    sW1 = -dW1;
    sb1 = -db1;
    sW2 = -dW2;
    sb2 = -db2;
end
