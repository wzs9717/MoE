function X_onehot=getTopN(X,n)
[~,arg_sort]=sort(X);
[m1,m2]=size(X);
% topN=arg_sort(end-n+1:end);
X_onehot=zeros(m1,m2);
X_onehot(arg_sort(end-n+1:end))=1;
end