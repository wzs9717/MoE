function Y_sort_ind=get_distributions(num_catch)
% -------------accurance statistic----------
[~,arg_sort]=sort(num_catch);
arg_sort=arg_sort(9:10,:);
save('arg_sort_3.mat','arg_sort');
load('arg_sort_3.mat')
X_onehot=zeros(10,t_end);
for i=1:t_end
X_onehot(arg_sort(:,i),i)=1;
end
X=twoTen(X_onehot);
[Y_sort,Y_sort_ind]=hist_distribution(X);
plot(Y_sort);
end