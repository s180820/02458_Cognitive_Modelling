from scipy.optimize import minimize
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class Variance_Model:
    "lalallalaall"
    def __init__(self, n_experiments, n_trials, mu_s, mu_s0, sigma_s, sigma_s0):
        self.n_experiments = n_experiments
        self.n_trials = n_trials
        self.mu_s = mu_s
        self.mu_s0 = mu_s0
        self.sigma_s = sigma_s
        self.sigma_s0 = sigma_s0
    
    def simulate(self, cs):
        d_prime_unb = []
        d_prime_b_yes = []
        d_prime_b_no = []

        for c in cs:
            
            for i in range(100):
                n_tp = sum(np.random.normal(self.mu_s, self.sigma_s, (self.n_trials)) > c)
                n_fp = sum(np.random.normal(self.mu_s0, self.sigma_s0, (self.n_trials)) > c)

                if c == cs[0]:
                    d_prime_unb.append(norm.ppf(n_tp/self.n_trials)-norm.ppf(n_fp/self.n_trials))
                elif c == cs[1]:
                    d_prime_b_yes.append(norm.ppf(n_tp/self.n_trials)-norm.ppf(n_fp/self.n_trials))
                elif c == cs[2]:
                    d_prime_b_no.append(norm.ppf(n_tp/self.n_trials)-norm.ppf(n_fp/self.n_trials))
        
        return d_prime_unb, d_prime_b_yes, d_prime_b_no

    def plot_histogram(self, cs):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        d_prime_unb, d_prime_b_yes, d_prime_b_no = self.simulate(cs)

        ax[0].hist(d_prime_unb, bins=30)
        ax[0].set_title("Unbiased - d'")

        ax[1].hist(d_prime_b_yes, bins=30)
        ax[1].set_title("Biased towards yes - d'")

        ax[2].hist(d_prime_b_no, bins=30)
        ax[2].set_title("Biased towards no - d'")

        plt.show()
        print("The d prime for the unbiased criterion:", np.mean(d_prime_unb))
        print("The d prime for the biased towards yes criterion:", np.mean(d_prime_b_yes))
        print("The d prime for the biased towards no criterion:", np.mean(d_prime_b_no))
        

    



class PsychoMetric:
    def __init__(self, stimulus_intensity: np.array, number_of_correct_responses: np.array, number_of_trial: int, p_guess=None) -> None:
        self.si = stimulus_intensity
        self.nr = number_of_trial
        self.nc = number_of_correct_responses
        self.p_guess = p_guess


    def NLL_psycho(self, parameters, model):
        c = parameters[0]
        sigma = parameters[1]
        L = 0
        for i in range(len(self.si)):
            P_s = norm.cdf((self.si[i]-c)/sigma)
            if model == "HT":
                P_s = P_s + (1 - P_s)*self.p_guess
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
        res = minimize(self.NLL_psycho, initial_guess, model, method='L-BFGS-B')
        print("c = ", res.x[0])
        print("sigma = ", res.x[1])
        print("Negative log likelihood = ", res.fun)