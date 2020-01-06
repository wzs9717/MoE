function F2=sub_execution(link1,link2,I,a)
funNormalize = @(x) ( x-min(min(x(:))))/( max(max(x))-min(min(x)) + eps);
RS1=(sum(sum(link1,2)>0)/sum(sum(link1,1)>0))*(sum(link1(:))/sum(link1(:)>0));
RS2=sum(sum(link2,2)>0)/sum(sum(link2,1)>0)*1;
%------------------input----------------------------
    s=I;
    s=funNormalize((a*RS1).^(funNormalize(s)));
%------------------computing-----------------------    
    F1=s*link1;
    F1=funNormalize((a*RS2).^(funNormalize(F1)));
    
    F2=funNormalize(F1*link2);
end