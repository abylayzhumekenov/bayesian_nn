% Create a DNN with input size 2, hidden size 3, and output size 1
dnn = DNN(2, 3, 1);

% Define training data
x = [1, 2; 3, 4; 5, 6]'; % Input data (2 features, 3 samples)
y = [1, 0, 1];          % Target labels

% Train the network
dnn.train(x, y, 0.01, 100);