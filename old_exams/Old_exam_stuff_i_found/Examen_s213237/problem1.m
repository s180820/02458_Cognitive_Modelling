%% problem 1: late MLE model

% load data
Sub1=load('DataSub1.txt');
Sub2=load('DataSub2.txt');
Sub3=load('DataSub3.txt');
Sub4=load('DataSub4.txt');
Sub5=load('DataSub5.txt');

% late MLE model: 
% - observers base their responses to auditory and visual
% stimuli on an underlying auditory and visual continous internal
% representations SA and SV (early MLE)
% - audiovisual intergration is based in FLMP model

%% Question 1.3
% optimize the 4 free parameters to minimize the negative logarithm 
% of the likelihood between predicted probability and our observed data

% we initialize the free parameters
 
Sub=Sub1;
param0=[1 1 1 1];
fun=@(param)myfun(param,Sub);
paramf=fminunc(fun,param0);
% paramf returns sigmaA, sigmaV, cA and cV respectively
 
% And then we find Nlog with the fitted parameters
% PA and PV based on early MLE model
N=24;
sigmaA=exp(paramf(1))
sigmaV=exp(paramf(2))
cA=paramf(3)
cV=paramf(4)

x=1:5;
muA=x-cA;
muV=x-cV;
PA_i=normcdf(muA/(sigmaA));
PV_i=normcdf(muV/(sigmaV));

% PAV based on FLMP model
PAV_i=zeros(5,5);
for a=1:5
    for v=1:5
        PAV_i(v,a) = (PA_i(a).*PV_i(v))./((PA_i(a).*PV_i(v))+((1-PA_i(a)).*(1-PV_i(v))));
    end
end

pAVmatrix=[PA_i;PV_i;PAV_i];
for k=1:7
    pAVf(k,:)=binopdf(Sub(k,:),N,pAVmatrix(k,:));
end

Nlog=-log(prod(prod(pAVf)))
 
%% 1.4. Free parameters and max likelihood for each subject + Nlog
% we change the subject in the previous part by hand (no time to do a loop)

% Sub= Sub1, Sub2, Sub3, Sub4, Sub5 in the section above

% we take the values from the command window and add them in the results
% table





