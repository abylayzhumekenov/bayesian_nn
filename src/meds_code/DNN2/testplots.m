% --- 1. Generate High-Dimensional Synthetic Data ---
N = 500;   % Number of samples
n = 50;    % High-dimensional input features
m = 10;    % Output dimensions
D = 3;     % Number of hidden layers
hidden_sizes = [100, 50, 25];  % Hidden layer sizes

X = randn(N, n);  % High-dimensional input data
w_true = randn(n, m);  % True weights for linear mapping
Y = X * w_true + 0.1 * randn(N, m);  % Linear function with noise

% --- 2. Initialize Deep Neural Network Parameters ---
A = cell(D+1, 1);
B = cell(D+1, 1);
b = cell(D+1, 1);

layer_sizes = [n, hidden_sizes, m];

for d = 1:D+1
    A{d} = randn(layer_sizes(d+1), layer_sizes(d)) * sqrt(2 / (layer_sizes(d) + layer_sizes(d+1)));
    B{d} = ones(layer_sizes(d+1), layer_sizes(d));  % Variance (uncertainty measure)
    b{d} = zeros(1, layer_sizes(d+1));
end

% --- 3. Activation and Loss Functions ---
relu = @(z) max(0, z);
mse_loss = @(Y_pred, Y) mean(sum((Y_pred - Y).^2, 2));  % Ensuring scalar loss

% --- 4. Train Using Martingale Posterior Updates ---
epochs = 1000;
learning_rate = 0.01;
alpha = 0.9;  % Confidence-weighted update factor

A_vals = cell(D+1, 1);
b_vals = cell(D+1, 1);
loss_vals = zeros(epochs, 1);

for d = 1:D+1
    A_vals{d} = zeros([size(A{d}), epochs]);
    b_vals{d} = zeros([size(b{d}), epochs]);
end

for epoch = 1:epochs
    % Forward pass
    H = X;
    activations = cell(D+1, 1);
    for d = 1:D
        H = relu(H * A{d}' + repmat(b{d}, N, 1));
        activations{d} = H;
    end
    Y_pred = H * A{D+1}' + repmat(b{D+1}, N, 1);
    
    % Compute loss
    loss = mse_loss(Y_pred, Y);
    loss_vals(epoch) = loss;
    
    % Store parameter evolution
    for d = 1:D+1
        A_vals{d}(:,:,epoch) = A{d};
        b_vals{d}(:,:,epoch) = b{d};
    end
    
    % Compute gradients
    dY = 2 * (Y_pred - Y) / N;
    dA = cell(D+1, 1);
    db = cell(D+1, 1);
    
    % Backpropagation
    dH = dY;
    for d = D+1:-1:2
        dA{d} = dH' * activations{d-1};
        db{d} = sum(dH, 1);
        dH = (dH * A{d}) .* (activations{d-1} > 0);
    end
    dA{1} = dH' * X;
    db{1} = sum(dH, 1);
    
    % Bayesian Posterior Update (Martingale)
    for d = 1:D+1
        B{d} = alpha * B{d} + (1 - alpha) * abs(dA{d});
        A{d} = A{d} - learning_rate * dA{d} ./ (B{d} + 1e-6);
        b{d} = b{d} - learning_rate * db{d};
    end
    
    if mod(epoch, 100) == 0
        disp(['Epoch: ' num2str(epoch) ', Loss: ' num2str(loss)]);
    end
end

% --- 5. Plot Results ---
figure;
plot(1:epochs, loss_vals);
xlabel('Epoch');
ylabel('Loss (MSE)');
title('Loss Evolution Over Time');
grid on;

figure;
for d = 1:D+1
    subplot(D+1,1,d);
    plot(squeeze(mean(A_vals{d}, [1,2])));
    xlabel('Epoch');
    ylabel(['Mean A' num2str(25*d) ' Values']);
    title(['Evolution of Posterior Mean A' num2str(25*d)]);
    grid on;
end

figure;
for d = 1:D+1
    subplot(D+1,1,d);
    plot(squeeze(mean(b_vals{d}, 2)));
    xlabel('Epoch');
    ylabel(['Mean b' num2str(25*d) ' Values']);
    title(['Evolution of Bias b' num2str(25*d)]);
    grid on;
end
