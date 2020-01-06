function [sublink1,sublink2]=subModelTrain(subTrainsetsX,subTrainsetsY,thresh)
s_net=50;%size of network unit
data_train_Y=[0,1,2,3,4,5,6,7,8,9]*subTrainsetsY';
%-------------------------preprocess trainsets--------------------------------------
%-------------------------preprocess testsets--------------------------------------
% ----------------creat the input and output trainset-------------------
[m1, m2, m] = size(subTrainsetsX);
t_end=m;
m3=m1*m2;
I1=reshape(subTrainsetsX,m3,m)';%input layer
I2=zeros(1,s_net);%middle layer
I3=subTrainsetsY;%output layer
%--------------------initialise parameters---------------------------
% batch=0;
num=1;%the number of connected cell 
t=1;%number of iteration
link1=zeros(m3,s_net);
link2=zeros(s_net,10);
catch_train_accu=[];
catch_test_accu=[];
catch_iter_accu=[];
catch_nums=[];
%% main
%-------------------- LOOP----------------------------------
while t<=t_end
%-------------enlarge link matrix every s_net steps------------------------   
    [~,s_net_now]=size(link1);
    if num>s_net_now
        s_net_now
        link1=[link1,zeros(m3,s_net)];
        link2=[link2;zeros(s_net,10)];
        I2=[I2,zeros(1,s_net)];
    end
%-------------input------------------------------------
    s1=I1(t,:);
    s2=I2;
    s2(num)=1;
    s3=I3(t,:);
 %-----------precheck------------------------------
    error=MINIST_check(s1,data_train_Y(t),link1,link2,thresh);%precheck if the network can recognize the image
    if not(error)
        t=t+1;
    else 
%         catch_nums=[catch_nums,t];
%-------------update link matrix--------------------    
        link1=link_update(link1,s1,s2);
        link2=link_update(link2,s2,s3);
        t=t+1;
        num=num+1;
    end
%---------print t--------------------------------------    
     if rem(t,1000)==0
         t
     end
end
sublink1=link1;
sublink2=link2;
end