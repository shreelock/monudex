p = predict(Theta1, Theta2, tsurf_feat);
k=p(:,2);
h=zeros(5,1);
ggi=zeros(5,1);
k1=sort(k,'descend');
for i=1:5
    h(i)=find(k==k1(i))
end
k2=sort(k,'ascend');
for i=1:5
    gg=find(k==k2(i));
    ggi(i)=gg(1)
end