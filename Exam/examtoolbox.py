from scipy.optimize import minimize
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline
from scipy.optimize import curve_fit

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

    def return_parameters(self, initial_guess, model):
        res = minimize(self.NLL_psycho, initial_guess, model, method='L-BFGS-B')
        return res.x[0], res.x[1]

    def plot_psycho(self, initial_guess):
        c_ht, sigma_ht = self.return_parameters(initial_guess, "HT")
        c_psy, sigma_psy = self.return_parameters(initial_guess, "psycho")

        P_s_psy = []
        P_s_th = []
        for i in range(len(self.si)):
            P_spsy = norm.cdf((self.si[i]-c_psy)/sigma_psy)
            x = norm.cdf((self.si[i]-c_ht)/sigma_ht)
            P_sht = x + (1 - x)*self.p_guess
            P_s_psy.append(P_spsy)
            P_s_th.append(P_sht)
        

        # 300 represents number of points to make between T.min and T.max
        xnew = np.linspace(self.si.min(), self.si.max(), 300) 

        spl_psy = make_interp_spline(self.si, P_s_psy, k=3)
        spl_th = make_interp_spline(self.si, P_s_th, k=3)  # type: BSpline
        power_smooth_psy = spl_psy(xnew)
        power_smooth_th = spl_th(xnew)
        plt.figure(figsize=(15, 5))
        sns.lineplot(x=xnew, y=power_smooth_psy, label="Psycho")
        sns.lineplot(x=xnew, y=power_smooth_th, label="HT")
        sns.scatterplot(x=self.si, y=self.nc/self.nr, color="black", label="Observed Data")
        plt.show()

class MagnitudeEsimation:
    def __init__(self, number_of_stimuli: int, a: float) -> None:
        self.i_s = np.arange(1, number_of_stimuli +1) 
        self.a = a

    def stevens(self):
        return 10*self.i_s**(self.a)

    def fechner(self, i_s, c, I0):
        return (1/c)*np.log(i_s/I0)
    
    def print_fit(self):
        i_p = self.stevens()
        param, cov = curve_fit(self.fechner, self.i_s, i_p, maxfev=5000)
        fit_fech = self.fechner(self.i_s, param[0], param[1])

        print("Optimal Weber fraction", param[0])
        print("I0:", param[1])

        # plot the results
        plt.plot(self.i_s, i_p, 'o-',label='Steven\'s Law')
        plt.plot(self.i_s, fit_fech, 'o-', label='Fitted Fechner\'s Law (with a={})'.format(self.a))
        plt.xlabel('Physical Intensity')
        plt.ylabel('Perceived Intensity')
        #plt.title('Fitted Fechner\'s Law (with a=0.33)')
        plt.legend()
        plt.show()