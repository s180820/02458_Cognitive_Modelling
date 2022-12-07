
function mse = myfun(param,datasub)

N=24;
sigmaA=exp(param(1));
sigmaV=exp(param(2));
cA=param(3); 
cV=param(4); 

x=1:5;
muA=x-cA;
muV=x-cV;

% PA and PV based on early MLE model
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
    pAVf(k,:)=binopdf(datasub(k,:),N,pAVmatrix(k,:));
end

Nlog=-log(prod(prod(pAVf)));

end