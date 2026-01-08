clear; clc; 

% Define Neural Network Structure
D = 3;  % Number of layers (excluding input)
n = [10, 20, 10, 1]; % Example layer sizes [input, hidden1, hidden2, output]

% Initialize Weights and Biases
weights = cell(D, 1);
biases = cell(D, 1);

for d = 1:D
    weights{d} = randn(n(d+1), n(d));  % Weight matrix for layer d
    biases{d} = randn(n(d+1), 1);      % Bias vector for layer d
end

% Step size function
step_size = @(m) 1 / (m + 1);

% Number of training samples
N = 1000;

% Generate synthetic dataset (for demonstration purposes)
x_data = randn(n(1), N); % Input features
y_data = randn(n(end), N); % Target values for regression

% Set the number of iterations per martingale
T = 500;
D_martingales = 50; % Number of independent martingales

% Pre-allocate Theta_samples for parallel execution
Theta_samples = cell(D_martingales, 1);

%% Ensure Parallel Pool is Managed Properly
pool = gcp('nocreate'); % Check if a parallel pool exists
if isempty(pool)
    parpool;  % Start a new parallel pool if none exists
else
    disp('Parallel pool is already running.');
end

%% Martingale Posterior Sampling
parfor d = 1:D_martingales
    % Initialize independent copy of weights and biases for each martingale
    weights_m = cell(D, 1);
    biases_m = cell(D, 1);
    
    for layer = 1:D
        weights_m{layer} = weights{layer};
        biases_m{layer} = biases{layer};
    end
    
    % Martingale iterations
    for m = 1:T
        idx = randi(N);  % Pick a random data point
        x_m = x_data(:, idx);
        y_m = y_data(:, idx);
        
        % Compute forward pass
        [f_x, activations] = forward_pass(x_m, weights_m, biases_m, D);
        
        % Compute Score Function
        score = compute_score(y_m, f_x, weights_m, biases_m, activations, D);
        
        % Debugging: Ensure dimensions match
        for layer = 1:D
            if size(weights_m{layer}) == size(score{layer, 1})
                weights_m{layer} = weights_m{layer} + step_size(m) * score{layer, 1};  % Update weights
            else
                disp(['Mismatch in weights update at layer ', num2str(layer)]);
            end
            
            if size(biases_m{layer}) == size(score{layer, 2})
                biases_m{layer} = biases_m{layer} + step_size(m) * score{layer, 2};  % Update biases
            else
                disp(['Mismatch in biases update at layer ', num2str(layer)]);
            end
        end
    end
    
    % Store final parameters
    Theta_samples{d} = {weights_m, biases_m};
end

% Close parallel pool to free resources
delete(gcp('nocreate'));

%% Visualization of Weight and Bias Evolution
figure;
for d = 1:D
    subplot(2, D, d);
    hold on;
    for i = 1:D_martingales
        imagesc(Theta_samples{i}{1}{d});  % Display the weight matrices
        colorbar;
    end
    title(sprintf('Weights A_{%d}', d));
    hold off;
    
    subplot(2, D, d + D);
    hold on;
    for i = 1:D_martingales
        plot(Theta_samples{i}{2}{d}, 'LineWidth', 1.5);  % Display bias values
    end
    title(sprintf('Biases b_{%d}', d));
    hold off;
end

%% Function Definitions

% Forward Pass Function
function [f_x, activations] = forward_pass(x, weights, biases, D)
    activations = cell(D, 1);
    a = x;
    
    for d = 1:D-1
        a = relu(weights{d} * a + biases{d});  % Forward pass for hidden layers
        activations{d} = a;
    end
    
    % Final layer (linear activation)
    f_x = weights{D} * a + biases{D};  
    activations{D} = f_x;
end

% Compute Score Function (Gradient of Log-Likelihood)
function score = compute_score(y, f_x, weights, biases, activations, D)
    score = cell(D, 2);
    
    % Compute Gradient of Log-Likelihood (Regression Case)
    Sigma_inv = eye(length(y)); % Assuming identity covariance matrix
    grad_loss = Sigma_inv * (y - f_x); % Gradient w.r.t. output
    
    % Backpropagation of Gradients
    delta = grad_loss;
    for d = D:-1:1
        score{d, 1} = delta * activations{d}';  % dL/dA (gradient w.r.t weights)
        score{d, 2} = delta; % dL/db (gradient w.r.t biases)
        
        if d > 1
            delta = (weights{d}' * delta) .* (activations{d-1} > 0); % Backpropagation through ReLU
        end
    end
end

% ReLU Activation Function
function y = relu(x)
    y = max(0, x);
end
