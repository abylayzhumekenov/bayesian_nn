classdef DNN
    properties
        % Network parameters
        theta
    end
    
    methods
        function obj = DNN(inputSize, hiddenSize, outputSize)
            % Initialize network parameters
            obj.theta.W1 = dlarray(randn(hiddenSize, inputSize) * 0.01);
            obj.theta.b1 = dlarray(zeros(hiddenSize, 1));
            obj.theta.W2 = dlarray(randn(hiddenSize, hiddenSize) * 0.01);
            obj.theta.b2 = dlarray(zeros(hiddenSize, 1));
            obj.theta.W3 = dlarray(randn(outputSize, hiddenSize) * 0.01);
            obj.theta.b3 = dlarray(zeros(outputSize, 1));
        end
        
        function [y_pred, h1, h2] = predictNetwork(obj, x)
            % Forward pass through the network
            W1_data = extractdata(obj.theta.W1);
            x_data = extractdata(x);
            b1_data = extractdata(obj.theta.b1);
            
            % Assert to check that inputs are numeric
            assert(isnumeric(W1_data), 'W1 must be numeric');
            assert(isnumeric(x_data), 'x must be numeric');
            assert(isnumeric(b1_data), 'b1 must be numeric');
            
            h1 = relu(W1_data * x_data + b1_data);  % Use matrix multiplication
            h2 = relu(extractdata(obj.theta.W2) * h1 + extractdata(obj.theta.b2));
            y_pred = extractdata(obj.theta.W3) * h2 + extractdata(obj.theta.b3);
            
            % Ensure y_pred is a dlarray
            y_pred = dlarray(y_pred);
        end
        
        function [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3] = computeGradients(obj, x_dl, y_dl, y_pred, h1, h2)
            % Ensure y_pred and y are valid inputs
            assert(~isempty(y_pred), 'y_pred must not be empty');
            assert(~isempty(y_dl), 'y must not be empty');
            assert(isa(y_pred, 'dlarray') || isnumeric(y_pred), 'y_pred must be a dlarray or numeric');
            assert(isa(y_dl, 'dlarray') || isnumeric(y_dl), 'y must be a dlarray or numeric');
            
            % Extract data from y_pred and y
            y_pred_data = extractdata(y_pred);
            y_data = extractdata(y_dl);
            
            % Compute the gradient
            dL_dy = 2 * (y_pred_data - y_data); % Use extractdata to remove dimension labels
            
            % Backpropagation
            dL_dh2 = obj.theta.W3' * dL_dy;
            dL_dW3 = dL_dy * h2';
            dL_db3 = dL_dy;
            
            dL_dh1 = obj.theta.W2' * (dL_dh2 .* (h2 > 0));
            dL_dW2 = (dL_dh2 .* (h2 > 0)) * h1';
            dL_db2 = dL_dh2 .* (h2 > 0);
            
            dL_dx = obj.theta.W1' * (dL_dh1 .* (h1 > 0));
            dL_dW1 = (dL_dh1 .* (h1 > 0)) * extractdata(x_dl)';
            dL_db1 = dL_dh1 .* (h1 > 0);
            
            % Return gradients
            grad_W1 = dL_dW1;
            grad_b1 = dL_db1;
            grad_W2 = dL_dW2;
            grad_b2 = dL_db2;
            grad_W3 = dL_dW3;
            grad_b3 = dL_db3;
        end
        
        function train(obj, x, y, learningRate, numEpochs)
            % Convert inputs to dlarray
            x_dl = dlarray(x);
            y_dl = dlarray(y);
            
            for epoch = 1:numEpochs
                % Forward pass
                [y_pred, h1, h2] = obj.predictNetwork(x_dl);
                
                % Compute gradients
                [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3] = obj.computeGradients(x_dl, y_dl, y_pred, h1, h2);
                
                % Update parameters
                obj.theta.W1 = obj.theta.W1 - learningRate * grad_W1;
                obj.theta.b1 = obj.theta.b1 - learningRate * grad_b1;
                obj.theta.W2 = obj.theta.W2 - learningRate * grad_W2;
                obj.theta.b2 = obj.theta.b2 - learningRate * grad_b2;
                obj.theta.W3 = obj.theta.W3 - learningRate * grad_W3;
                obj.theta.b3 = obj.theta.b3 - learningRate * grad_b3;
                
                % Display loss
                loss = mean((extractdata(y_pred) - extractdata(y_dl)).^2);
                fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
            end
        end
    end
end

% Helper function for ReLU activation
function y = relu(x)
    y = max(0, x);
end