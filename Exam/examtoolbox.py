from scipy.optimize import minimize
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

class Variance_Model:
    """
    Function to use for the equal and unequal variance model. If sigma_s0 is not equal to sigma_s, 
    the model is unequal variance.

    input:
        n_experiments: number of experiments to simulate (int)
        n_trials: number of trials per experiment (int)
        mu_s: mean of signal distribution (float)
        mu_s0: mean of noise distribution (float)
        sigma_s: standard deviation of signal distribution (float)
        sigma_s0: standard deviation of noise distribution (float)

    parameters:
        simulate(self, cs): simulates the model for the given parameters and returns the d' for the three different criteria
        plot_histogram(self, cs): plots the histogram of the d' for the three different criteria and prints the d'
    """
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
        
class Variance_Model_2:
    """
    Function to use for the unequal variance model with multiple criterions.

    input:
        n_experiments: number of experiments to simulate (int)
        n_trials: number of trials per experiment (int)
        n_participants: number of participants (int)
        cs: list of criterions (list)
    
    parameters:
        simulate(self): simulates the model for the given criterions and outputs a list of mus and sigmas
        plot_histogram(self): plots the histogram of mus and sigmas from the simulate function
    """
    def __init__(self, n_experiments, n_trials, n_participants, cs):
        self.n_exp = n_experiments
        self.n_trials = n_trials
        self.n_subjects = n_participants
        self.low_c = cs[0]
        self.mid_c = cs[1]
        self.high_c = cs[2]

    def simulate(self):
        sigma_list = []
        mu_list = []
        x_es = []
        y_es = []
        for i in range(self.n_exp):
            stim_choices = np.random.normal(1, 0.8,50)
            no_stim_choices = np.random.normal(0, 1 ,50)

            yes_high_s =    sum([1 if i >= self.high_c else 0 for i in stim_choices])
            yes_low_s =     sum([1 if i < self.high_c and i > self.mid_c else 0 for i in stim_choices])
            no_low_s =      sum([1 if i > self.low_c and i <= self.mid_c else 0 for i in stim_choices])
            no_high_s =     sum([1 if i <= self.low_c else 0 for i in stim_choices])

            yes_high_s0 =   sum([1 if i >= self.high_c else 0 for i in no_stim_choices])
            yes_low_s0 =    sum([1 if i < self.high_c and i > self.mid_c else 0 for i in no_stim_choices])
            no_low_s0 =     sum([1 if i > self.low_c and i <= self.mid_c else 0 for i in no_stim_choices])
            no_high_s0 =    sum([1 if i <= self.low_c else 0 for i in no_stim_choices])


            tp_c1 = yes_high_s /self.n_trials
            tp_c2 = (yes_high_s + yes_low_s) /self.n_trials
            tp_c3 = (yes_high_s + yes_low_s + no_low_s) /self.n_trials


            fp_c1 = yes_high_s0 /self.n_trials
            fp_c2 = (yes_high_s0 + yes_low_s0) /self.n_trials
            fp_c3 = (yes_high_s0 + yes_low_s0 + no_low_s0) /self.n_trials

            y= np.array([norm.ppf(tp_c1), norm.ppf(tp_c2), norm.ppf(tp_c3)])
            x= np.array([norm.ppf(fp_c1), norm.ppf(fp_c2), norm.ppf(fp_c3)]).reshape((-1,1))

            x_es.append(x)
            y_es.append(y)
            
            model = LinearRegression().fit(x, y)
            intercept, slope = model.intercept_, model.coef_[0]

            sigma = 1/slope
            mu_ses = sigma*intercept

            sigma_list.append(sigma)
            mu_list.append(mu_ses)
        return mu_list, sigma_list

    def plot_histogram(self):
        mu_list, sigma_list = self.simulate()
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        colors =["skyblue", "olive", "teal"]
        sns.distplot(mu_list, color=colors[0], ax=ax[0,0])
        sns.lineplot(x=1, y=[0,2], linewidth=2,color='red', ax=ax[0,0], label='True mu')
        ax[0,0].set_title('mu')
        sns.histplot(mu_list, color=colors[0], ax=ax[0,1])
        sns.lineplot(x=1, y=[0,20], linewidth=2,color='red', ax=ax[0,1], label='True mu')
        ax[0,1].set_title('mu')
        sns.distplot(sigma_list, color=colors[1], ax=ax[1,0])
        sns.lineplot(x=0.8, y=[0,2], linewidth=2,color='red', ax=ax[1,0], label='True sigma')
        ax[1,0].set_title('sigma')
        sns.histplot(sigma_list, color=colors[1], ax=ax[1,1])
        sns.lineplot(x=0.8, y=[0,2], linewidth=2,color='red', ax=ax[1,1], label='True sigma')
        ax[1,1].set_title('sigma')
        plt.show()

class PsychoMetric:
    """
    Function to use for the psychometric function.

    input:
        stimulus_intensity: list of stimulus intensities (np.arrray)
        number_of_correct_responses: list of number of correct responses (np.array)
        number_of_trials: number of trials (int)
        p_guess: guess rate (float) (only needed for high threshold model)

    parameters:
        NLL_psycho(self, paramters, model): calculates the negative log likelihood for the psychometric function given the parameters and the model
        print_parameters(self, initial_guess, model): prints the parameters of the model
        return_parameters(self, initial_guess, model): returns the parameters of the model
        plot_psycho(self, initial_guess): plots the psychometric functions for both models
    """
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
    """
    Function to use for the magnitude estimation function.

    input:
        number_of_stimuli: number of stimuli intensities (int)
        a: parameter a (float)
    
    parameters:
        stenvens(): calculates the Stevens function from intensties and a
        fechners(i_s, c, I0): calculates the Fechner function from intensities, c and I0
        print_fit(): plots the fitted Stevens and Fechners law of the data
    """
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
        plt.legend()
        plt.show()