function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];
C=X*(Theta1');
c=sigmoid(C);
q = size(c, 1);
c = [ones(q, 1) c];
D=c*(Theta2');
d=sigmoid(D);
Y=zeros(m,num_labels);
for i=1:m
    Y(i,y(i))=1;
end
for i=1:m
    for k=1:num_labels
  J=J-(Y(i,k)*log(d(i,k)))-((1-Y(i,k))*log(1-d(i,k)));      
    end
end
x=Theta1 .^ 2;
x(:,1)=zeros(size(x,1),1);
z=Theta2 .^ 2;
z(:,1)=zeros(size(z,1),1);
J=J+ (lambda/2)*(sum(sum(x))+sum(sum(z)));
J=J/m;

d3=d-Y;
C=[ones(size(C,1),1),C];
 d2=(d3*Theta2).*sigmoidGradient(C);
 
a2=c;
a3=d;
a1=X;

A= (a1')*d2;
A=A(:,2:end);

I=lambda*Theta1/m;
I(:,1)=0;
Theta1_grad=Theta1_grad+(A')/m+I;
B= (a2)'*d3;

I=lambda*Theta2/m;
I(:,1)=0;
Theta2_grad=Theta2_grad+(B')/m+I;

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
