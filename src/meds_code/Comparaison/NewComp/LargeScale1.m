% martingale_vs_smc_mnist_large.m
% Compare Martingale Posterior vs SMC on MNIST
% with a large‐scale 5‐layer net 

clear; close all; clc;

%% 1. Load & preprocess MNIST
[XTrain4D, YTrainCat] = digitTrain4DArrayData;  % 5000 images
[XTest4D,  YTestCat ] = digitTest4DArrayData;   % 10000 images

% Flatten to N×784, labels 0:9
Xall   = reshape(XTrain4D, [28*28, size(XTrain4D,4)])';
Yall   = grp2idx(YTrainCat) - 1;
XtestA = reshape(XTest4D,  [28*28, size(XTest4D,4)])';
YtestA = grp2idx(YTestCat)  - 1;

% Desired subsample sizes (use full MNIST if available)
reqTrain = 50000;
reqTest  = 10000;

nAllTrain = size(Xall,1);
nAllTest  = size(XtestA,1);
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

%% 2. Large network architecture (5‐layer FC net)
D = size(Xtrain,2);
dims = [ D, 1024, 512, 256, 128, K ];  % 5 layers: 784→1024→512→256→128→10

%% 3. Martingale Posterior
alpha     = 1e-4;    % prior precision
batchSize = 256;
rng(1);
tic;
[meanMP, varMP] = martingalePosterior(Xtrain, Ytrain_1hot, dims, alpha, batchSize);
timeMP = toc;
fprintf('Martingale Posterior completed in %.2f s\n', timeMP);

%% 4. Sequential Monte Carlo
N        = 200;      % particles
ess_thr  = N/2;
batchSize = 256;
rng(1);
tic;
[meanSMC, varSMC] = smcPosterior(Xtrain, Ytrain_1hot, dims, N, ess_thr, batchSize);
timeSMC = toc;
fprintf('SMC completed in %.2f s\n', timeSMC);

%% 5. Evaluate
Kmc   = 50;
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
  P     = sum( dims(1:end-1).*dims(2:end) + dims(2:end) );
  S     = zeros(P,1);
  Q     = alpha*ones(P,1);
  theta = zeros(P,1);
  Ndata = size(X,1);
  idx   = randperm(Ndata);
  for t = 1:ceil(Ndata/batchSize)
    b = idx((t-1)*batchSize+1 : min(t*batchSize,Ndata));
    x = X(b,:)'; y = Y1h(:,b);
    [~, grad] = netLossGrad(theta, x, y, dims);
    score    = -grad;
    S        = S + score;
    Q        = Q + score.^2;
    theta    = S ./ Q;
  end
  mTheta = theta;
  vTheta = 1 ./ Q;
end

function [mTheta, vTheta] = smcPosterior(X, Y1h, dims, N, ess_thr, batchSize)
  P      = sum( dims(1:end-1).*dims(2:end) + dims(2:end) );
  thetas = repmat(initializeTheta(dims),1,N) + 0.1*randn(P,N);
  logw   = zeros(N,1);
  Ndata  = size(X,1);
  idx    = randperm(Ndata);
  for t = 1:ceil(Ndata/batchSize)
    b = idx((t-1)*batchSize+1 : min(t*batchSize,Ndata));
    x = X(b,:)'; y = Y1h(:,b);
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
  L = numel(dims)-1;
  % Unpack
  W = cell(L,1); b = cell(L,1);
  idx = 0;
  for l = 1:L
    Dp = dims(l); Dn = dims(l+1);
    W{l} = reshape(theta(idx+1:idx+Dp*Dn), Dn, Dp);
    idx = idx + Dp*Dn;
    b{l} = theta(idx+1:idx+Dn);
    idx = idx + Dn;
  end
  N = size(X,2);
  % Forward
  A = cell(L+1,1); Z = cell(L,1);
  A{1} = X;
  for l = 1:L-1
    Z{l} = W{l}*A{l} + b{l};
    A{l+1} = max(0, Z{l});
  end
  Z{L} = W{L}*A{L} + b{L};
  P    = softmax(Z{L});
  loss = -sum(log(sum(Y1h .* P,1))) / N;
  % Backward
  G    = cell(L,1);
  gradW = cell(L,1);
  gradb = cell(L,1);
  G{L}    = (P - Y1h) / N;
  gradW{L} = G{L} * A{L}';
  gradb{L} = sum(G{L},2);
  for l = L-1:-1:1
    dA = W{l+1}' * G{l+1};
    G{l} = dA .* (Z{l} > 0);
    gradW{l} = G{l} * A{l}';
    gradb{l} = sum(G{l},2);
  end
  % Pack
  grad = [];
  for l = 1:L
    grad = [grad; gradW{l}(:); gradb{l}];
  end
end

function Z = forwardPass(theta, X, dims)
  L = numel(dims)-1;
  W = cell(L,1); b = cell(L,1);
  idx = 0;
  for l = 1:L
    Dp = dims(l); Dn = dims(l+1);
    W{l} = reshape(theta(idx+1:idx+Dp*Dn), Dn, Dp);
    idx = idx + Dp*Dn;
    b{l} = theta(idx+1:idx+Dn);
    idx = idx + Dn;
  end
  A = X;
  for l = 1:L-1
    A = max(0, W{l}*A + b{l});
  end
  Z = W{L}*A + b{L};
end

function theta = initializeTheta(dims)
  L = numel(dims)-1;
  theta = [];
  for l = 1:L
    W = 0.1*randn(dims(l+1), dims(l));
    b = zeros(dims(l+1),1);
    theta = [theta; W(:); b];
  end
end

function Y = softmax(X)
  E = exp(X - max(X,[],1));
  Y = E ./ sum(E,1);
end

function idx = resampleMultinomial(w, N)
  edges = [0; cumsum(w(:))];
  edges(end) = 1;
  u = rand(N,1);
  idx = arrayfun(@(x) find(x>edges(1:end-1)&x<=edges(2:end),1), u);
end