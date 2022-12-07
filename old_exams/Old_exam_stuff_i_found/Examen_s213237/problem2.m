%% problem 2

data=[4 15 22 7 2; 2 4 8 10 26];

HITs=zeros(1,4);
FAs=zeros(1,4);
for i=1:4
    HIT=(sum(data(2,:))-sum(data(2,1:i)))/50
    FA=(sum(data(1,:))-sum(data(1,1:i)))/50
    HITs(i)=HIT;
    FAs(i)=FA;
end

figure(1)
scatter(norminv(FAs),norminv(HITs))
figure(2)
plot(norminv(FAs),norminv(HITs))

% let's fit a linear model to the ROC in the gaussian space
mld=fitlm(norminv(FAs(1:end)),norminv(HITs(1:end)));
% from that we obtain the intercept and the slope
inter=table2array(mld.Coefficients(1,1));
slope=table2array(mld.Coefficients(2,1));
% and we use it to compute sigma and mu assuming unequal SDT
sigma=1/slope
mu=inter/slope

%% Rebin data: pool categories 2-5 into a single one
% calculate d'
pool1=[4 46; 2 48];

HIT1=(50-2)/50
FA1=(50-4)/50
% From P(Hit)=P(FA)/sigma + mu/sigma

mu=sigma*(HIT1-FA1/sigma)

%% Rebin data: pool categories 1-4 into a single one
% calculate d'
pool2=[48 2; 24 26];

HIT2=(50-24)/50;
FA2=(50-48)/50;

mu=sigma*(HIT2-FA2/sigma)




