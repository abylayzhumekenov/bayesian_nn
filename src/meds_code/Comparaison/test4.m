% Martingale Posterior Sampling for DNN with PDF and Histogram Plots
clear; clc; close all; rng(1);

%% Simulated Data
N = 100; x = linspace(-3, 3, N)';
y = sin(x) + 0.3*randn(N,1);

%% Network Architecture
d_in = 1; d_hidden = 10; d_out = 1;
f = @(x, theta) theta.W2 * tanh(theta.W1 * x' + theta.b1) + theta.b2;

%% Parameters
T = 200;       % Updates per sample
D = 100;       % Number of posterior samples

%% Martingale Posterior Sampling
theta0 = initParams(d_in, d_hidden, d_out);
posterior_samples = cell(D,1);
for d = 1:D
    theta = theta0;
    for t = 1:T
        idx = randi(N); x_t = x(idx); y_t = y(idx);
        [sW1, sb1, sW2, sb2] = scoreFunc(x_t, y_t, theta);
        eta = 1/(t + 1);
        theta.W1 = theta.W1 + eta * sW1;
        theta.b1 = theta.b1 + eta * sb1;
        theta.W2 = theta.W2 + eta * sW2;
        theta.b2 = theta.b2 + eta * sb2;
    end
    posterior_samples{d} = theta;
end

%% Extract samples to matrices
W1_all = cell2mat(cellfun(@(th) th.W1(:)', posterior_samples, 'UniformOutput', false));
b1_all = cell2mat(cellfun(@(th) th.b1(:)', posterior_samples, 'UniformOutput', false));
W2_all = cell2mat(cellfun(@(th) th.W2(:)', posterior_samples, 'UniformOutput', false));
b2_all = cell2mat(cellfun(@(th) th.b2(:)', posterior_samples, 'UniformOutput', false));

%% Plot PDFs + Histograms
plotPDFHist(W1_all, 'W1', 5);
plotPDFHist(b1_all, 'b1', 5);
plotPDFHist(W2_all, 'W2', 1);
plotPDFHist(b2_all, 'b2', 1);

%% Save as PDF
set(gcf, 'PaperPositionMode','auto');
saveas(gcf, 'Martingale_Posterior_Plots.pdf');

%% --- FUNCTIONS ---

function theta = initParams(din, dh, dout)
    theta.W1 = 0.1*randn(dh, din);
    theta.b1 = 0.1*randn(dh,1);
    theta.W2 = 0.1*randn(dout, dh);
    theta.b2 = 0.1*randn(dout,1);
end

function [sW1, sb1, sW2, sb2] = scoreFunc(x, y, theta)
    a1 = theta.W1 * x + theta.b1;
    h = tanh(a1);
    y_hat = theta.W2 * h + theta.b2;
    res = y - y_hat;
    dy = -res;
    dW2 = dy * h';
    db2 = dy;
    dh = theta.W2' * dy;
    da1 = dh .* (1 - tanh(a1).^2);
    dW1 = da1 * x';
    db1 = da1;
    sW1 = -dW1; sb1 = -db1; sW2 = -dW2; sb2 = -db2;
end

function plotPDFHist(samples, name, max_plots)
    num_params = size(samples, 2);
    figure('Name', ['Martingale Posterior: ', name], 'Color', 'w');
    for i = 1:min(max_plots, num_params)
        subplot(ceil(max_plots/2), 2, i);
        histogram(samples(:,i), 'Normalization','pdf','FaceAlpha',0.4);
        hold on;
        [f, xi] = ksdensity(samples(:,i));
        plot(xi, f, 'r-', 'LineWidth', 2);
        title([name, '(', num2str(i), ')']);
        xlabel('Value'); ylabel('PDF');
        grid on;
    end
end
