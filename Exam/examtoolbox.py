from scipy.optimize import minimize
import numpy as np
from scipy.stats import norm



class PsychoMetric:
    def __init__(self, stimulus_intensity: np.array, number_of_correct_responses: np.array, number_of_trial: int) -> None:
        self.si = stimulus_intensity
        self.nr = number_of_trial
        self.nc = number_of_correct_responses


    def NLL_psycho(self, parameters):
        c = parameters[0]
        sigma = parameters[1]
        L = 0
        for i in range(len(self.si)):
            P_s = norm.cdf((self.si[i]-c)/sigma)
            log_NS = 0
            log_ns = 0
            log_Nns = 0
            for j in range(self.nr):
                log_NS += np.log(j+1)
            for j in range(self.nc[i]):
                log_ns += np.log(j+1)
            for j in range(self.nr - self.nc[i]):
                log_Nns += np.log(j+1)
            if self.nr - self.nc[i] == 0:
                L += log_NS - log_ns - log_Nns + self.nc[i]*np.log(P_s)
            else:
                L += log_NS - log_ns - log_Nns + self.nc[i]*np.log(P_s) + (self.nr - self.nc[i]) * np.log(1 - P_s)
        return -L
    
    def print_parameters(self, initial_guess, model):
        if model == "psycho":
            res = minimize(self.NLL_psycho, initial_guess, method='L-BFGS-B')
            print("c = ", res.x[0])
            print("sigma = ", res.x[1])
            print("Negative log likelihood = ", res.fun)