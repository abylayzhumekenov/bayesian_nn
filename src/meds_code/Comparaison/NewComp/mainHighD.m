% martingale_vs_smc_mnist_medium_fixed.m
% Compare Martingale Posterior vs SMC on MNIST
% with a medium-sized 2-layer net
% This version auto‐clips nTrain/nTest to the dataset size.

clear; close all; clc;

%% 1. Load & preprocess MNIST
[XTrain4D, YTrainCat] = digitTrain4DArrayData;  % returns 5000 images
[XTest4D,  YTestCat ] = digitTest4DArrayData;   % returns 10000 images

% Flatten to N×784, labels 0:9
Xall   = reshape(XTrain4D, [28*28, size(XTrain4D,4)])';
Yall   = grp2idx(YTrainCat) - 1;
XtestA = reshape(XTest4D,  [28*28, size(XTest4D,4)])';
YtestA = grp2idx(YTestCat)  - 1;

% Desired subsample sizes
reqTrain = 20000;
reqTest  = 5000;

% Actual sizes
nAllTrain = size(Xall,1);
nAllTest  = size(XtestA,1);

% Clip
nTrain = min(reqTrain, nAllTrain);
nTest  = min(reqTest,  nAllTest);

fprintf('Using %d / %d train images, %d / %d test images\n', ...
        nTrain, nAllTrain, nTest, nAllTest);

Xtrain = Xall(1:nTrain,:);
Ytrain = Yall(1:nTrain);
Xtest  = XtestA(1:nTest,:);
Ytest  = YtestA(1:nTest);

% One‐hot encode training labels
K = 10;
Ytrain_1hot = full(ind2vec(Ytrain'+1, K));

%% 2. Network architecture (medium)
D = size(Xtrain,2);
H = 300;             % hidden units
dims = [D, H, K];    % [784,300,10]

%% 3. Martingale Posterior
alpha     = 1e-3;    % prior precision
batchSize = 128;
rng(0);
tic;
[meanMP, varMP] = martingalePosterior(Xtrain, Ytrain_1hot, dims, alpha, batchSize);
timeMP = toc;
fprintf('Martingale Posterior completed in %.2f s\n', timeMP);

%% 4. Sequential Monte Carlo
N        = 100;      % particles
ess_thr  = N/2;
rng(0);
tic;
[meanSMC, varSMC] = smcPosterior(Xtrain, Ytrain_1hot, dims, N, ess_thr, batchSize);
timeSMC = toc;
fprintf('SMC completed in %.2f s\n', timeSMC);

%% 5. Evaluate
Kmc   = 30;
accMP = predictAndEvaluateMP(Xtest, Ytest, dims, meanMP, varMP, Kmc);
accSMC= predictAndEvaluateSMC(Xtest, Ytest, dims, meanSMC);
fprintf('Test Accuracy: MP = %.2f%%,  SMC = %.2f%%\n', accMP*100, accSMC*100);

%% 6. Plot runtimes
figure('Position',[200 200 500 300]);
bar([timeMP, timeSMC]);
set(gca,'XTickLabel',{'Martingale','SMC'});
ylabel('Time (s)');
title('Runtime Comparison');


%% ===== Nested Functions =====

function [mTheta, vTheta] = martingalePosterior(X, Y1h, dims, alpha, batchSize)
  P     = numel(initializeTheta(dims));
  S     = zeros(P,1);
  Q     = alpha*ones(P,1);
  theta = zeros(P,1);
  Ndata = size(X,1);
  idx   = randperm(Ndata);
  for t = 1:ceil(Ndata/batchSize)
    b = idx((t-1)*batchSize+1 : min(t*batchSize,Ndata));
    x = X(b,:)';  y = Y1h(:,b);
    [~, grad] = netLossGrad(theta, x, y, dims);
    score    = -grad;      % Fisher score
    S        = S + score;
    Q        = Q + score.^2;
    theta    = S ./ Q;
  end
  mTheta = theta;
  vTheta = 1 ./ Q;
end

function [mTheta, vTheta] = smcPosterior(X, Y1h, dims, N, ess_thr, batchSize)
  P      = numel(initializeTheta(dims));
  thetas = repmat(initializeTheta(dims),1,N) + 1e-1*randn(P,N);
  logw   = zeros(N,1);
  Ndata  = size(X,1);
  idx    = randperm(Ndata);
  for t = 1:ceil(Ndata/batchSize)
    b = idx((t-1)*batchSize+1 : min(t*batchSize,Ndata));
    x = X(b,:)';  y = Y1h(:,b);
    for i = 1:N
      [loss, ~] = netLossGrad(thetas(:,i), x, y, dims);
      logw(i)   = logw(i) - loss;
    end
    w = exp(logw - max(logw));
    w = w / sum(w);
    if 1/sum(w.^2) < ess_thr
      thetas = thetas(:, resampleMultinomial(w, N));
      logw   = zeros(N,1);
    end
  end
  mTheta = mean(thetas,2);
  vTheta = var(thetas,0,2);
end

function acc = predictAndEvaluateMP(X, Y, dims, mTheta, vTheta, Kmc)
  M    = size(X,1);
  sumP = zeros(M, dims(end));
  for k = 1:Kmc
    theta_k = mTheta + sqrt(vTheta).*randn(size(mTheta));
    Z       = forwardPass(theta_k, X', dims);  % K×M
    Pmat    = softmax(Z);
    sumP    = sumP + Pmat';
  end
  [~, preds] = max(sumP / Kmc, [], 2);
  acc = mean(preds-1 == Y);
end

function acc = predictAndEvaluateSMC(X, Y, dims, mTheta)
  Z = forwardPass(mTheta, X', dims);  % K×M
  [~, preds] = max(Z,[],1);
  acc = mean((preds-1)' == Y);
end

function [loss, grad] = netLossGrad(theta, X, Y1h, dims)
  D = dims(1); H = dims(2); K = dims(3);
  % unpack
  idx = 0;
  W1  = reshape(theta(idx+1:idx+H*D), H, D); idx=idx+H*D;
  b1  = theta(idx+1:idx+H);                  idx=idx+H;
  W2  = reshape(theta(idx+1:idx+K*H), K, H); idx=idx+K*H;
  b2  = theta(idx+1:idx+K);
  N   = size(X,2);
  % forward
  Z1 = W1*X + b1;
  A1 = max(0, Z1);
  Z2 = W2*A1 + b2;
  P  = softmax(Z2);
  loss = -sum(log(sum(Y1h .* P,1))) / N;
  % backward
  G2  = (P - Y1h) / N;
  gW2 = G2 * A1';
  gb2 = sum(G2,2);
  G1  = (W2' * G2) .* (Z1 > 0);
  gW1 = G1 * X';
  gb1 = sum(G1,2);
  grad = [gW1(:); gb1; gW2(:); gb2];
end

function Z = forwardPass(theta, X, dims)
  D = dims(1); H = dims(2); K = dims(3);
  idx = 0;
  W1  = reshape(theta(idx+1:idx+H*D), H, D); idx=idx+H*D;
  b1  = theta(idx+1:idx+H);                  idx=idx+H;
  W2  = reshape(theta(idx+1:idx+K*H), K, H); idx=idx+K*H;
  b2  = theta(idx+1:idx+K);
  A1  = max(0, W1*X + b1);
  Z   = W2*A1 + b2;
end

function theta = initializeTheta(dims)
  D = dims(1); H = dims(2); K = dims(3);
  W1 = 0.1*randn(H,D); b1 = zeros(H,1);
  W2 = 0.1*randn(K,H); b2 = zeros(K,1);
  theta = [W1(:); b1; W2(:); b2];
end

function Y = softmax(X)
  E = exp(X - max(X,[],1));
  Y = E ./ sum(E,1);
end

function idx = resampleMultinomial(w, N)
  % Multinomial resampling without toolboxes
  edges = [0; cumsum(w(:))];
  edges(end) = 1;  % guard
  u = rand(N,1);
  idx = arrayfun(@(x) find(x>edges(1:end-1) & x<=edges(2:end),1), u);
end