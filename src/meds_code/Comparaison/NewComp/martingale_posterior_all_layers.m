function [w_samps,b_samps,elapsed] = martingale_posterior_all_layers(net,X,sigma2,D,T)
% martingale_posterior_all_layers  Run D independent martingale‐posterior chains
% over *all* parameters of a pretrained dlnetwork, each for T synthetic‐data updates.
% Returns D samples of the *last* element of the final fully‐connected layer’s
% weight‐matrix and bias‐vector, plus total wall‐clock time.
%
% Inputs:
%   net     – a dlnetwork object
%   X       – p×n matrix of n input feature‐vectors (columns)
%   sigma2  – assumed Gaussian noise variance (scalar)
%   D       – number of parallel chains (posterior samples)
%   T       – chain length (number of synthetic updates per chain)
%
% Outputs:
%   w_samps – D×1 samples of W_final(end,end)
%   b_samps – D×1 samples of b_final(end)
%   elapsed – total wall‐clock time in seconds

  n = size(X,2);

  % locate final-layer Weights & Bias in net.Learnables
  P = net.Learnables.Parameter;
  Widx = find(strcmp(P,'Weights'));
  Bidx = find(strcmp(P,'Bias'));
  lastW = Widx(end);
  lastB = Bidx(end);

  w_samps = zeros(D,1);
  b_samps = zeros(D,1);

  tic;
  parfor chain = 1:D
    % clone network so each chain is independent
    netChain = clone(net);

    for m = 1:T
      % 1) random training sample
      i   = randi(n);
      xdl = dlarray(X(:,i),'CB');    % channel×batch

      % 2) forward pass
      yPred = forward(netChain, xdl);
      yPred = extractdata(yPred);

      % 3) synthetic response
      y = yPred + sqrt(sigma2)*randn(size(yPred));
      ydl = dlarray(y,'CB');

      % 4) get gradients of loss = ‖yPred−y‖²/(2σ²)
      [grads,state] = dlfeval(@modelGradients, netChain, xdl, ydl, sigma2);
      if ~isempty(state)
        netChain.State = state;
      end

      % 5) martingale‐posterior update: θ ← θ − εₘ ∇L,  εₘ=1/(n+m)
      epsm = 1/(n + m);
      netChain.Learnables = dlupdate(@(p,g) p - epsm*g, netChain.Learnables, grads);
    end

    % extract last‐layer parameters
    L = netChain.Learnables;
    W = extractdata(L.Value{lastW});
    b = extractdata(L.Value{lastB});

    w_samps(chain) = W(end,end);
    b_samps(chain) = b(end);
  end
  elapsed = toc;

  % display summary
  fprintf('Completed %d chains × %d steps in %.2f sec\n',D,T,elapsed);
  fprintf(' Posterior W(end,end): mean=%.4f  std=%.4f\n',mean(w_samps),std(w_samps));
  fprintf(' Posterior b(end)   : mean=%.4f  std=%.4f\n',mean(b_samps),std(b_samps));

  % optional histograms
  figure;
  subplot(1,2,1), histogram(w_samps,40);
    title('W(end,end) Posterior'), xlabel('value');
  subplot(1,2,2), histogram(b_samps,40);
    title('b(end) Posterior'),      xlabel('value');
end

%----------------------------------------------------------------------
function [gradients, state] = modelGradients(net, x, y, sigma2)
  % Forward (capture state if any)
  try
    [yPred, state] = forward(net, x);
  catch
    yPred = forward(net, x);
    state = [];
  end
  % loss = ½‖y−yPred‖²/σ²
  loss = sum((yPred - y).^2,'all')/(2*sigma2);
  % back‐propagate to all learnable parameters
  gradients = dlgradient(loss, net.Learnables);
end