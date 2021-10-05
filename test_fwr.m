function [d1] = test_fwr(X,W,M1,M2,D,V,sw)

C1r = W'*X'*X*W;
T1 = log(diag(C1r));
T1 = T1.*sw';
T1 = V'*T1;
d1 = sum((T1-M1).^2./D);
d1 = d1-sum((T1-M2).^2./D);

end
