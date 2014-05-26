tic
close all; 

%% Setup the parameters you will use for this exercise
input_layer_size  = 2560;   %LENGTH OF SURF FEATURE MATRIX USED FOR EACH IMAGE [40x64]
hidden_layer_size = 25;   % 25 hidden units
num_labels = 5;          % 5 LABELS , SINCE WE NEED TO CLASSIFY IT INTO 5 GROUPS
                          

% Part 1: Loading  DatA
fprintf('Loading Data ...\n')
% THE DATABASE SHOULD HAVE surf_feat AND Grpvc 
% ELSE NEED TO  RUN  generate_surf_data.m BEFORE RUNNING nn4monu.m
X=surf_feat;
y=Grpvc;                %the matrix for initialising values to the images
m = size(X, 1);
Theta1=zeros(25,(size(X,2)+1));
Theta2=zeros(5,26);

fprintf('Data Loaded..\n');

nn_params = [Theta1(:) ; Theta2(:)];

% -- COMPUTING THE COST -- %
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(('Cost using given parameters: %f \n'), J);

% -- Checking the Regularisation -- %
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost NOw, using given parameters : %f \n'], J);
g = sigmoidGradient([1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);


initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
checkNNGradients;

lambda = 3;
checkNNGradients(lambda);

debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f \n\n'], debug_J);

% TRAINIG THE NETWORK
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50);  %LETS

lambda = 1;

costFunction = @(p) nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),num_labels, (hidden_layer_size + 1));


pred = predict(Theta1, Theta2, tsurf_feat);
and=[checkMat ; pred'];
fprintf('First Column shows Image indices, Second Column shows the Result Predicted\n');
fprintf('Heres How to read it \n 1- India Gate \n 2- Lotus Temple \n 3-Taj Mahal \n 4-Qutub Minar \n 5-Red Fort \n');
disp(and');
toc