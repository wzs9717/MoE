function [Y_noSparse,Y_ind]=hist_distribution(X)
Y=zeros(size(X));
for i=1:max(X(:))
Y(i)=sum(X==i);
end
[Y_sort,Y_ind]=sort(Y);
Y_noSparse=Y_sort(Y_sort~=0);
Y_ind=Y_ind(end-length(Y_noSparse)+1:end);
% Y_sort=sort(Y_noSparse);
end