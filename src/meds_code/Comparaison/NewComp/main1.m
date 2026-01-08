% martingale_vs_smc_mnist.m
% Compare Martingale Posterior vs SMC for uncertainty quantification 
% in a small DNN on MNIST (using built-in digitTrain4DArrayData)
% Requires MATLAB R2018b+ and Deep Learning Toolbox

clear; close all; clc;

%% 1. Load and preprocess MNIST (built-in)
% 1.1 Load 4D arrays of training and test images
[XTrain4D, YTrainCat] = digitTrain4DArrayData;
[XTest4D,  YTestCat ] = digitTest4DArrayData;

% 1.2 Reshape to (N×784) and convert categorical labels to 0:9
XtrainAll = reshape(XTrain4D, [28*28, size(XTrain4D,4)])';
YtrainAll = grp2idx(YTrainCat) - 1;   % categorical 1:10 → 0:9

XtestAll  = reshape(XTest4D,  [28*28, size(XTest4D,4)])';
YtestAll  = grp2idx(YTestCat)  - 1;

% 1.3 Subsample for speed
nTrain = 5000;
nTest  = 1000;
Xtrain = XtrainAll(1:nTrain,:);
Ytrain = YtrainAll(1:nTrain);
Xtest  = XtestAll(1:nTest,:);
Ytest  = YtestAll(1:nTest);

% 1.4 One-hot encode training labels
K = 10;
Ytrain_1hot = full(ind2vec(Ytrain'+1, K));

%% 2. Define network architecture
D = size(Xtrain,2);     % input dim = 784
H = 50;                  % hidden units
dims = [D, H, K];        % network dims
% theta0 = initializeTheta(dims);  % not used below

%% 3. Martingale Posterior inference
alpha     = 1e-3;    % prior precision (diagonal)
batchSize = 100;     % mini-batch size
rng(0);
tic;
[meanMP, varMP] = martingalePosterior(Xtrain, Ytrain_1hot, dims, alpha, batchSize);
timeMP = toc;
fprintf('Martingale Posterior completed in %.2f s\n', timeMP);

%% 4. Sequential Monte Carlo inference
N         = 100;     % number of particles
ess_thresh= N/2;
rng(0);
tic;
[meanSMC, varSMC] = smcPosterior(Xtrain, Ytrain_1hot, dims, N, ess_thresh, batchSize);
timeSMC = toc;
fprintf('SMC completed in %.2f s\n', timeSMC);

%% 5. Evaluate predictive performance
Kmc   = 50;  % Monte Carlo draws for MP
accMP = predictAndEvaluateMP(Xtest, Ytest, dims, meanMP, varMP, Kmc);
[accSMC, ~] = predictAndEvaluateSMC(Xtest, Ytest, dims, meanSMC, varSMC);

fprintf('Test Accuracy: MP = %.2f%%, SMC = %.2f%%\n', accMP*100, accSMC*100);

%% 6. Plot results
figure('Position',[100 100 800 400]);
%subplot(1,2,1);
bar([timeMP, timeSMC]);
set(gca,'XTickLabel',{'Martingale','SMC'});
ylabel('Runtime (s)');
title('Computational Cost');

% subplot(1,2,2);
% weightsIdx = 1;  % index of a chosen weight
% histogram(sqrt(varMP(weightsIdx)),'Normalization','pdf','DisplayStyle','stairs','LineWidth',2);
% hold on;
% histogram(sqrt(varSMC(weightsIdx)),'Normalization','pdf','DisplayStyle','stairs','LineWidth',2);
% legend('MP std','SMC std');
% xlabel('Std dev of weight(1)');
% title('Posterior Uncertainty');

%% ===== Nested Functions =====

function [mean_theta, var_theta] = martingalePosterior(X, Y1h, dims, alpha, batchSize)
    P = numel(initializeTheta(dims));
    S = zeros(P,1);
    Q = alpha*ones(P,1);      
    theta = zeros(P,1);
    Ndata = size(X,1);
    idx = randperm(Ndata);
    for t = 1:ceil(Ndata/batchSize)
        batch = idx((t-1)*batchSize+1 : min(t*batchSize,Ndata));
        x = X(batch,:)'; y = Y1h(:,batch);
        [~, g] = netLossGrad(theta, x, y, dims);
        score = -g;  % Fisher score
        S = S + score;
        Q = Q + score.^2;
        theta = S ./ Q;
    end
    mean_theta = theta;
    var_theta  = 1./Q;
end

function [mean_theta, var_theta] = smcPosterior(X, Y1h, dims, N, ess_thr, batchSize)
    P = numel(initializeTheta(dims));
    thetas = repmat(initializeTheta(dims),1,N) + 1e-1*randn(P,N);
    logw   = zeros(N,1);
    Ndata  = size(X,1);
    idx    = randperm(Ndata);
    for t = 1:ceil(Ndata/batchSize)
        batch = idx((t-1)*batchSize+1 : min(t*batchSize,Ndata));
        x = X(batch,:)'; y = Y1h(:,batch);
        for i = 1:N
            [loss, ~] = netLossGrad(thetas(:,i), x, y, dims);
            logw(i) = logw(i) - loss;
        end
        % normalize weights
        w = exp(logw - max(logw));
        w = w/sum(w);
        ess = 1/sum(w.^2);
        if ess < ess_thr
            idxs = resampleMultinomial(w, N);
            thetas = thetas(:,idxs);
            logw   = zeros(N,1);
        end
    end
    mean_theta = mean(thetas,2);
    var_theta  = var(thetas,0,2);
end

function acc = predictAndEvaluateMP(X, Y, dims, m, v, Kmc)
    M = size(X,1);
    logits_sum = zeros(M,10);
    for k = 1:Kmc
        theta_k = m + sqrt(v).*randn(size(m));
        Z = forwardPass(theta_k, X', dims);   % K×M
        P = softmax(Z);                       % K×M
        logits_sum = logits_sum + P';
    end
    [~, preds] = max(logits_sum/Kmc, [], 2);
    acc = mean(preds-1 == Y);
end

function [acc, predVar] = predictAndEvaluateSMC(X, Y, dims, mean_theta, var_theta)
    Z = forwardPass(mean_theta, X', dims);  % K×M
    [~, preds] = max(Z,[],1);
    acc = mean((preds-1)' == Y);
    predVar = var_theta;
end

function [loss, grad] = netLossGrad(theta, X, Y1h, dims)
    [W1,b1,W2,b2] = unpack(theta,dims);
    N = size(X,2);
    Z1 = W1*X + b1;
    A1 = max(0, Z1);
    Z2 = W2*A1 + b2;
    P  = softmax(Z2);
    loss = -sum(log(sum(Y1h.*P)))/N;
    G2 = (P - Y1h)/N;
    gW2 = G2 * A1';
    gb2 = sum(G2,2);
    G1  = (W2'*G2) .* (Z1>0);
    gW1 = G1 * X';
    gb1 = sum(G1,2);
    grad = pack(gW1,gb1,gW2,gb2);
end

function Z = forwardPass(theta, X, dims)
    [W1,b1,W2,b2] = unpack(theta,dims);
    A1 = max(0, W1*X + b1);
    Z  = W2*A1 + b2;
end

function theta = initializeTheta(dims)
    D = dims(1); H = dims(2); K = dims(3);
    W1 = 0.1*randn(H,D); b1 = zeros(H,1);
    W2 = 0.1*randn(K,H); b2 = zeros(K,1);
    theta = pack(W1,b1,W2,b2);
end

function v = pack(W1,b1,W2,b2)
    v = [W1(:); b1; W2(:); b2];
end

function [W1,b1,W2,b2] = unpack(theta,dims)
    D = dims(1); H = dims(2); K = dims(3);
    idx = 0;
    W1 = reshape(theta(idx+1:idx+H*D), H, D); idx = idx+H*D;
    b1 = theta(idx+1:idx+H);               idx = idx+H;
    W2 = reshape(theta(idx+1:idx+H*K), K, H); idx = idx+H*K;
    b2 = theta(idx+1:idx+K);
end

function Y = softmax(X)
    e = exp(X - max(X,[],1));
    Y = e ./ sum(e,1);
end

function idx = resampleMultinomial(w, N)
    edges = min([0; cumsum(w(:))], 1);
    edges(end) = 1;
    u1 = rand/N;
    [~, idx] = histc(u1:1/N:1, edges);
end