function g = sigmoidGradient(z)
%COMPUTES THE GRADIENT OF SIGMOID FUNCTION
g = zeros(size(z));
s=sigmoid(z);
[m n]=size(z);
for i=1:m
for j=1:n
    g(i,j)=s(i,j)*(1-s(i,j));
end
end

end
