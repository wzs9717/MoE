%% get distribution
addpath('data\'); 
addpath('matlab_fun\'); 
load('num_catch.mat')
% -------------accurance statistic----------
[~,arg_sort]=sort(num_catch');
arg_sort=arg_sort(9:10,:);
% save('data\arg_sort_3.mat','arg_sort');
% load('data\arg_sort_3.mat')
X_onehot=zeros(10,t_end);
t_end=60000;
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
% dic_sublink1 = containers.Map;
% dic_sublink2 = containers.Map;
% 
% %------------train sub models--------------------------------
% for i=1:20
%     [subTrainsetsX,subTrainsetsY]=buildSets(data_train_X_orig,data_train_Y_preprocessed,selector(i,:));
%     i
%     [sublink1,sublink2]=subModelTrain(subTrainsetsX,subTrainsetsY,thresh);
%     dic_sublink1(num2str(i))=sublink1;
%     dic_sublink2(num2str(i))=sublink2;
% end
% save('dic_sublink3.mat');
%--------------------------------end------------------------