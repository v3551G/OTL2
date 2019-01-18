function  EObeta
% EObeta: Effect of beta
%--------------------------------------------------------------------------
% No input
%
% Output: six figures for the effect of beta on the six datasets
%--------------------------------------------------------------------------

Beta_OTL = [];
ERROR_OTL = [];

Datasets = {'books_dvd','dvd_books','ele_kit','kit_ele','landmine1','landmine2'};

for ix =1:length(Datasets),

%load dataset
load(sprintf('data/%s',Datasets{1,ix}));
[n,d]       = size(data);
% set parameters
options.C   = 5;

%% set parameters: 'sigma'( kernel width) and 't_tick'(step size for plotting figures)
options.sigma = 4;
options.sigma2 = 8;
options.t_tick= round(length(ID_new)/15);

%%  
m = length(ID_new);
options.Number_old=n-m;
%ID_old = 1:n-m;
Y=data(1:n,1);
Y=full(Y);
X = data(1:n,2:d);


%% scale
MaxX=max(X,[],2);
MinX=min(X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
X = bsxfun(@minus, X, MinX);
X = bsxfun(@rdivide, X , DifX_2);

P = sum(X.*X,2);
P = full(P);
disp('Pre-computing kernel matrix...');
K = exp(-(repmat(P',n,1) + repmat(P,1,n)- 2*X*X')/(2*options.sigma^2));
% K = X*X';

X2 = X(n-m+1:n,:);
Y2 = Y(n-m+1:n);
P2 = sum(X2.*X2,2);
P2 = full(P2);
K2 = exp(-(repmat(P2',m,1) + repmat(P2,1,m)- 2*X2*X2')/(2*options.sigma2^2));
% K2 = X2*X2';
%% learn the old classifier
[h, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = avePA1_K_M(Y, K, options, ID_old);

fprintf(1,'The old classifier has %f support vectors\n',length(h.SV));

Beta = [2.^[-10:-1],1-2.^[-2:-1:-10]];
for j=1:length(Beta),
    options.beta = Beta(j);
    
    %% run experiments:
    for i=1:size(ID_new,1),
        fprintf(1,'running on the %d-th trial...\n',i);
        ID = ID_new(i, :);
        
        %5. OTL-II
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = OTL2_K_M(Y,K,K2,options,ID,h);
        nSV_OTL2(i) = length(classifier.SV1)+length(classifier.SV2);
        err_OTL2(i) = err_count;
        time_OTL2(i) = run_time;
        mistakes_list_OTL2(i,:) = mistakes;
        SVs_OTL2(i,:) = SVs;
        TMs_OTL2(i,:) = TMs;
    end
    ERR_OTL(j)= mean(err_OTL2/m)*100;
end

Beta_OTL=[Beta_OTL;Beta];
ERROR_OTL=[ERROR_OTL; ERR_OTL];

end

figure

plot(Beta_OTL(1,:),ERROR_OTL(1,:),'k-*');
hold on
plot(Beta_OTL(2,:),ERROR_OTL(2,:),'k-+');
plot(Beta_OTL(3,:),ERROR_OTL(3,:),'b-x');
plot(Beta_OTL(4,:),ERROR_OTL(4,:),'b-o');
plot(Beta_OTL(5,:),ERROR_OTL(5,:),'r-v');
plot(Beta_OTL(6,:),ERROR_OTL(6,:),'r-d');
legend('books-dvd','dvd-books','ele-kit','kit-ele','landmine1','landmine2');
xlabel('$beta$');
ylabel('Average rate of mistakes')
grid

