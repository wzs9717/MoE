function train_sub(,num_s)
%-------------get selector------------------------------
%num_s=40%number of sub models
selector=tenTwo(Y_sort_ind(end-num_s+1:end));
save('selector.mat','selector');
% clear data_train_X_tem;
% clear data_train_X;
% clear data_train_X_preprossed;
% clear data_train_X_orig;
% clear data_train_Y_preprocessed;
% sublink1=zeros(2,2,2);sublink2=zeros(2,2,2);
dic_sublink1 = containers.Map;
dic_sublink2 = containers.Map;

%------------train sub models--------------------------------
for i=1:20
    [subTrainsetsX,subTrainsetsY]=buildSets(data_train_X_orig,data_train_Y_preprocessed,selector(i,:));
    i
    [sublink1,sublink2]=subModelTrain(subTrainsetsX,subTrainsetsY,thresh);
    dic_sublink1(num2str(i))=sublink1;
    dic_sublink2(num2str(i))=sublink2;
end
save('dic_sublink3.mat');
end