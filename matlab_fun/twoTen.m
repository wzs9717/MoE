function X=twoTen(X_onehot)
X=X_onehot(1,:).*2^(0)+X_onehot(2,:).*2^(1)+X_onehot(3,:).*2^(2)+X_onehot(4,:).*2^(3)+X_onehot(5,:).*2^(4)+X_onehot(6,:).*2^(5)...
    +X_onehot(7,:).*2^(6)+X_onehot(8,:).*2^(7)+X_onehot(9,:).*2^(8)+X_onehot(10,:).*2^(9);
end