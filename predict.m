function p = predict(Theta1, Theta2, X)
%  Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);


p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2); %THE ONE WITH THE MAXIMUM WEIGHT IS THE SUPPOSED ANSWER
% p=h2;  % THIS VALUE WAS RETURNED FOR THE ON THE SPOT CODING PROBLEM
% =========================================================================


end
