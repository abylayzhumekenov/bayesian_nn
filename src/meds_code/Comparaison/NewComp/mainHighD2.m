clear; close all; clc;

%% 1. Load & preprocess MNIST
[XTrain4D, YTrainCat] = digitTrain4DArrayData;
[XTest4D,  YTestCat ] = digitTest4DArrayData;

% Flatten to N×784, labels 0:9
Xall   = reshape(XTrain4D, [28*28, size(XTrain4D,4)])';
Yall   = grp2idx(YTrainCat) - 1;
XtestA = reshape(XTest4D,  [28*28, size(XTest4D,4)])';
YtestA = grp2idx(YTestCat) - 1;

% Use subset for faster timing
nTrain = min(2000, size(Xall,1));
nTest  = min(1000, size(XtestA,1));
XtrainFull = Xall(1:nTrain,:);
Ytrain = Yall(1:nTrain);
XtestFull = XtestA(1:nTest,:);
Ytest = YtestA(1:nTest);

% One-hot encode
K = 10;
Ytrain_1hot = full(ind2vec(Ytrain'+1, K));

%% 2. Experiment setup
dims_list = [ 300, 400, 500, 600, 784];
H = 300;
batchSize = 128;
alpha = 1e-3;
N = 100;
ess_thr = N/2;

timeMP = zeros(size(dims_list));
timeSMC = zeros(size(dims_list));

%% 3. Loop over dimensions
for i = 1:length(dims_list)
    D = dims_list(i);
    dims = [D, H, K];

    % Truncate inputs
    Xtrain = XtrainFull(:, 1:D);
    Xtest  = XtestFull(:, 1:D);

    fprintf('\n=== Dimension: %d ===\n', D);

    % Martingale Posterior
    rng(0);
    tic;
    [meanMP, varMP] = martingalePosterior(Xtrain, Ytrain_1hot, dims, alpha, batchSize);
    timeMP(i) = toc;
    fprintf('Martingale time: %.2f s\n', timeMP(i));

    % SMC Posterior
    rng(0);
    tic;
    [meanSMC, varSMC] = smcPosterior(Xtrain, Ytrain_1hot, dims, N, ess_thr, batchSize);
    timeSMC(i) = toc;
    fprintf('SMC time: %.2f s\n', timeSMC(i));
end

%% 4. Plot as grouped histogram
figure('Position',[200 200 700 400]);

data = [timeMP(:), timeSMC(:)];
bar(dims_list, data, 'grouped');
xlabel('Input Dimension D');
ylabel('CPU Time (seconds)');
legend('Martingale', 'SMC', 'Location', 'northwest');
title('Runtime Comparison (Grouped Histogram)');
grid on;


%% ========== Helper Functions ==========

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
    score    = -grad;
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

function [loss, grad] = netLossGrad(theta, X, Y1h, dims)
  D = dims(1); H = dims(2); K = dims(3);
  idx = 0;
  W1  = reshape(theta(idx+1:idx+H*D), H, D); idx=idx+H*D;
  b1  = theta(idx+1:idx+H);                  idx=idx+H;
  W2  = reshape(theta(idx+1:idx+K*H), K, H); idx=idx+K*H;
  b2  = theta(idx+1:idx+K);
  N   = size(X,2);
  Z1 = W1*X + b1;
  A1 = max(0, Z1);
  Z2 = W2*A1 + b2;
  P  = softmax(Z2);
  loss = -sum(log(sum(Y1h .* P,1))) / N;
  G2  = (P - Y1h) / N;
  gW2 = G2 * A1';
  gb2 = sum(G2,2);
  G1  = (W2' * G2) .* (Z1 > 0);
  gW1 = G1 * X';
  gb1 = sum(G1,2);
  grad = [gW1(:); gb1; gW2(:); gb2];
end

function theta = initializeTheta(dims)
  D = dims(1); H = dims(2); K = dims(3);
  W1 = 0.1*randn(H,D); b1 = zeros(H,1);
  W2 = 0.1*randn(K,H); b2 = zeros(K,1);
  theta = [W1(:); b1; W2(:); b2];
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

function Y = softmax(X)
  E = exp(X - max(X,[],1));
  Y = E ./ sum(E,1);
end

function idx = resampleMultinomial(w, N)
  edges = [0; cumsum(w(:))];
  edges(end) = 1;  % guard
  u = rand(N,1);
  idx = arrayfun(@(x) find(x>edges(1:end-1) & x<=edges(2:end),1), u);
end
