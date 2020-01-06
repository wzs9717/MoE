function [subTrainsetsX,subTrainsetsY]=buildSets(data_train_X_orig,data_train_Y_preprocessed,selector)
funNormalize = @(x) ( x-min(min(x(:))))/( max(max(x))-min(min(x)) + eps);
ind=selector*data_train_Y_preprocessed';
ind_s=find(ind~=0);
subTrainsetsY=data_train_Y_preprocessed(ind_s,:);
%subTrainsetsX=data_train_X_preprossed(:,:,ind~=0);
data_train_X=permute(data_train_X_orig,[2,1,3]);
for i=1:length(ind_s)
data_train_X_tem=preprocess(data_train_X(:,:,ind_s(i)));
subTrainsetsX(:,:,i)= double(funNormalize(data_train_X_tem));
end

end