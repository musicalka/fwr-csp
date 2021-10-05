function [W,M1,M2,D,V,sw] = train_fwr(EEG,m,sigma)

% extract data for all epochs of the first class concatenated (EEG{1}) and 
% all epochs of the second class concatenated (EEG{2})
%m and sigma - parameter for regularization

D1=EEG{1};
D2=EEG{2};
[n, ch, tr] = size(D1); [n, ch, tr2] = size(D2);
C1 = zeros(ch,ch); C2 = C1;

for i=1:tr
    Cr1 =  D1(:,:,i)'*D1(:,:,i);
    C1 = C1+Cr1./trace(Cr1);
end

for i=1:tr2
    Cr2 = D2(:,:,i)'*D2(:,:,i);
    C2 = C2+Cr2./trace(Cr2);
end

C1 = C1/tr; C2=C2/tr2;

[W, E] = eig(inv(C1+C2)*C1); 
E = diag(E);
E = abs(E-0.5); [E, Ei] = sort(E, 'descend');
W = W(:,Ei);

for i=1:tr
    C1r(:,:,i) = W'*D1(:,:,i)'*D1(:,:,i)*W;
    F1(:,i) = log(diag(C1r(:,:,i)));
end
for i=1:tr2
    C2r(:,:,i) = W'*D2(:,:,i)'*D2(:,:,i)*W;
    F2(:,i) = log(diag(C2r(:,:,i)));
end

u=0:1:ch-1;
sw=exp(-u.^2/(2*(sigma).^2));

F1 = F1.*repmat(sw',1,tr); F2 = F2.*repmat(sw',1,tr2);

M1 = (mean(F1'))'; M2 = (mean(F2'))';
C1 = cov(F1'); C2 = cov(F2'); C = C1+C2;
[V, D] = eig(C); D = diag(D); 
[D, Di] = sort(D, 'descend'); V = V(:,Di);


a = D(1)*D(m)*(m-1)/(D(1)-D(m));
b = (m*D(m)-D(1))/(D(1)-D(m));

D(1:m) = D(1:m)+D(ch/2);
D(m+1:ch) = a./((m+1:1:ch)+b)+D(ch/2);

M1 = V'*M1; M2 = V'*M2;

