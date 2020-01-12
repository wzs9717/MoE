%% get distribution
addpath('data\'); 
addpath('matlab_fun\'); 
load('num_catch.mat')
% -------------accurance statistic----------
[~,arg_sort]=sort(num_catch');
arg_sort=arg_sort(9:10,:);
% save('data\arg_sort_3.mat','arg_sort');
% load('data\arg_sort_3.mat')
t_end=10000;
X_onehot=zeros(10,t_end);

for i=1:t_end
X_onehot(arg_sort(:,i),i)=1;
end
X=twoTen(X_onehot);
[Y_sort,Y_sort_ind]=hist_distribution(X);
plot(Y_sort);
%% train selector and submodels
%-------------get selector------------------------------
num_s=5%number of sub models
selector=tenTwo(Y_sort_ind(end-num_s+1:end));
save('data\selector.mat','selector');