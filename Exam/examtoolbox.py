import os
import cv2
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import scipy
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import make_interp_spline, BSpline

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

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
    def __init__(self, n_experiments, n_trials, n_participants, sigma_s, sigma_s0, cs):
        self.n_exp = n_experiments
        self.n_trials = n_trials
        self.n_subjects = n_participants
        self.low_c = cs[0]
        self.mid_c = cs[1]
        self.high_c = cs[2]
        self.sigma_s = sigma_s
        self.sigma_s0 = sigma_s0

    def simulate(self, return_tp_fp=False):
        sigma_list = []
        mu_list = []
        x_es = []
        y_es = []
        tps = []
        fps = []
        tns = []
        fns = []
        for i in range(self.n_exp):
            stim_choices = np.random.normal(1, self.sigma_s,self.n_trials)
            no_stim_choices = np.random.normal(0, self.sigma_s0 ,self.n_trials)

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

            tn_c1 = (no_high_s + no_low_s + yes_low_s) /self.n_trials
            tn_c2 = (no_high_s + no_low_s) /self.n_trials
            tn_c3 = (no_high_s) /self.n_trials

            fp_c1 = yes_high_s0 /self.n_trials
            fp_c2 = (yes_high_s0 + yes_low_s0) /self.n_trials
            fp_c3 = (yes_high_s0 + yes_low_s0 + no_low_s0) /self.n_trials

            fn_c1 = (no_high_s0 + no_low_s0 + yes_low_s0) /self.n_trials
            fn_c2 = (no_high_s0 + no_low_s0) /self.n_trials
            fn_c3 = (no_high_s0) /self.n_trials

            y= np.array([norm.ppf(tp_c1), norm.ppf(tp_c2), norm.ppf(tp_c3)])
            x= np.array([norm.ppf(fp_c1), norm.ppf(fp_c2), norm.ppf(fp_c3)]).reshape((-1,1))

            x_es.append(x)
            y_es.append(y)
            
            tps.append(np.array([tp_c1, tp_c2, tp_c3]))
            fps.append(np.array([fp_c1, fp_c2, fp_c3]))
            tns.append(np.array([tn_c1, tn_c2, tn_c3]))
            fns.append(np.array([fn_c1, fn_c2, fn_c3]))

            model = LinearRegression().fit(x, y)
            intercept, slope = model.intercept_, model.coef_[0]

            sigma = 1/slope
            mu_ses = sigma*intercept

            sigma_list.append(sigma)
            mu_list.append(mu_ses)

        if return_tp_fp:
            tps = [sum(sub_list) / len(sub_list) for sub_list in zip(*tps)]
            fps = [sum(sub_list) / len(sub_list) for sub_list in zip(*fps)]
            tns = [sum(sub_list) / len(sub_list) for sub_list in zip(*tns)]
            fns = [sum(sub_list) / len(sub_list) for sub_list in zip(*fns)]
            return tps, fps, tns, fns

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
        plt.xlabel("Stimulus Intensity")
        plt.ylabel("Probability of Correct Response")
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

class PCA_Images:
    def __init__(self, path) -> None:
        self.path = path

    def load_images(self):
        images = []
        for filename in os.listdir(self.path):
            img = cv2.imread(self.path + filename)
            if img is not None:
                print(self.path + filename)
                images.append(self.img_to_grayscale(np.array(img)))
        return images
    
    def img_to_grayscale(self, im):
        x, y, channels = list(im.shape)
        arr = np.empty((x, y, 1))
        for i, e in enumerate(np.array(im)):
            for j, l in enumerate(e):
                arr[i][j] = l[0]
        return arr

    def plot_images(self, images):
        for i in range(len(images)):
            plt.figure(figsize=(5, 5))
            plt.imshow(images[i], cmap="gray")
            plt.show()

    def get_mean_image(self, image_data, show=False):
        mean_image = np.mean(image_data, axis=(0))
        mean_image = mean_image.astype(int)
        if show==True:
            plt.figure(figsize=(5, 5))
            plt.imshow(mean_image, cmap="gray")
            plt.show()
        return mean_image
    
    def pca(self, image_data, n_components):
        self.n_components = n_components
        mean_image = self.get_mean_image(image_data)
        image_data = np.subtract(image_data, mean_image)
        for_pca = np.reshape(image_data, (image_data.shape[0], -1))
        print(for_pca.shape)
        pca = PCA(n_components)
        pc_scores = pca.fit_transform(for_pca)
        return pca, pc_scores

    def explained_variance(self, pca):
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Principal components',fontsize=14) 
        plt.ylabel('Variance explained',fontsize=14) 
        plt.xticks(fontsize=14); plt.yticks(fontsize = 14)
        plt.xlim(0,len(pca.explained_variance_ratio_)) 
        plt.ylim(0,1) 
        plt.grid()

        print("The first " + str(len(pca.explained_variance_ratio_)) + " components return " + str(np.sum(pca.explained_variance_ratio_)) + " of variance")

    def plot_pc(self, pca, pc_scores, mean_image):
        maximage=[]
        minimage=[]

        pcmax = pc_scores.max(0)
        pcmin = pc_scores.min(0)

        for i,col in enumerate(pc_scores.T):
            maximage.append(np.dot(pcmax[i],pca.components_[i]))
            minimage.append(np.dot(pcmin[i],pca.components_[i]))
            
        maximage_reshaped = np.reshape(maximage,(self.n_components,200,200,1))
        minimage_reshaped = np.reshape(minimage,(self.n_components,200,200,1))

        pca_visual = mean_image + maximage_reshaped - minimage_reshaped

        fig = plt.figure(figsize=(10, 5))
        for i,img in enumerate(pca_visual):
            plt.subplot(2,int(self.n_components/2),i + 1)
            axs = plt.imshow(pca_visual[i], cmap="gray")
            plt.title('PC'+str(i+1),fontsize = 5)
            axs.axes.get_xaxis().set_visible(False)
            axs.axes.get_yaxis().set_visible(False)

    
    def forward_selection(self, pc_scores, ratings, n_features, scoring):
        X = pc_scores
        y = ratings
        # Build RF classifier to use in feature selection
        #clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        clf = LinearRegression()

        # Build step forward feature selection
        sfs1 = sfs(clf,
                k_features=n_features,
                forward=True,
                floating=False,
                verbose=2,
                scoring=scoring,
                cv=10)

        sfs1 = sfs1.fit(X, y)
        # Which features?
        self.feat_cols = list(sfs1.k_feature_idx_)

        fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_err')
        plt.title('Sequential Forward Selection (w. StdERR)')
        plt.grid()
        plt.show()

    def linear_model(self, pca, pc_scores, ratings, mean_image):
        filtered_pc = []
        for i in range(len(pc_scores)):
            row = [pc_scores[i][j] for j in self.feat_cols]
            row = np.reshape(row, (len(self.feat_cols), )).T
            filtered_pc.append(row)

        filtered_components = []
        for i in range(len(self.feat_cols)):
            row = pca.components_[i]
            filtered_components.append(row)

        reg = LinearRegression()
        reg.fit(filtered_pc, ratings)

        synthetic_range = np.linspace(-0.2, 1.2, 5)
        alphas = []

        for i in synthetic_range:
            alphas.append((i - reg.intercept_) / (np.sum(np.abs(reg.coef_)**2)))

        zs = [alpha * reg.coef_ for alpha in alphas]
        new_imgs = [np.dot(z, filtered_components) for z in zs]

        plt.figure(figsize=(50, 30))
        for i,img in enumerate(new_imgs):
            plt.subplot(1,5,i + 1)
            fig = plt.imshow(np.reshape(img, (200, 200, 1)) + mean_image, cmap='gray')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

class MLE:
    def __init__(self, n_responses, data):
        self.n_responses = n_responses
        self.data = data
        self.x_a = data.iloc[0,:]
        self.x_v = data.iloc[1,:]
        self.x_av = data.iloc[2:,:]

    def sig(self, x):
        return 1/(1 + np.exp(-x))

    def cdf_a_v(self, mu, sigma, c):
        return norm.cdf(mu, c, sigma)

    def cdf_av(self, mu_a, mu_v, sigma_a, sigma_v, c_a, c_v):
        mu_av = (sigma_v ** 2 / (sigma_a ** 2 + sigma_v ** 2)) * (mu_a - c_a) + (sigma_a ** 2 / (sigma_a ** 2 + sigma_v ** 2)) * (mu_v - c_v)
        sigma_av = math.sqrt((sigma_a ** 2 * sigma_v ** 2) / (sigma_a ** 2 + sigma_v ** 2))
        return norm.cdf(mu_av/sigma_av)

    def gauss_likelihood_a_v(self, mu, sigma, c, x):
        return scipy.special.binom(self.n_responses, x) * self.cdf_a_v(mu, sigma, c)**x * (1 - self.cdf_a_v(mu, sigma, c))**(self.n_responses - x)

    def gauss_likelihood_av(self, mu_a, mu_v, sigma_a, sigma_v, c_a, c_v, x):
        return scipy.special.binom(self.n_responses, x) * self.cdf_av(mu_a, mu_v, sigma_a, sigma_v, c_a, c_v) ** x * (1 - self.cdf_av(mu_a, mu_v, sigma_a, sigma_v, c_a, c_v)) ** (self.n_responses - x)

    def log_likelihood_gaussian(self, params, d):
        c_a, c_v, sigma_a, sigma_v = params
        sigma_a = np.exp(sigma_a)
        sigma_v = np.exp(sigma_v)
        l_av = []
        l_a = np.prod([self.gauss_likelihood_a_v(j + 1, sigma_a, c_a, d[0][j]) for j in range(0, 5)])
        l_v = np.prod([self.gauss_likelihood_a_v(j + 1, sigma_v, c_v, d[1][j]) for j in range(0, 5)])
        
        for i in range(0, 5):
            for j in range(0, 5):
                l_av.append(self.gauss_likelihood_av(j + 1, i + 1, sigma_a, sigma_v, c_a, c_v, d[i+2][j]))
                
        return -np.log(np.prod([l_a, l_v, np.prod(l_av)]))

    def softmax(self, theta):
        return (np.exp(theta) / (np.exp(theta) + 1))

    def binom_likelihood_a_v(self, theta, x):
        return scipy.special.binom(self.n_responses, x) * self.softmax(theta)**x * (1 - self.softmax(theta))**(self.n_responses - x)

    def binom_likelihood_av(self, thetas, x):
        return scipy.special.binom(self.n_responses, x) * self.p_av(thetas) ** x * (1 - self.p_av(thetas)) ** (self.n_responses - x) 

    def p_av(self, thetas):
        return (self.softmax(thetas[0]) * self.softmax(thetas[1])) / (self.softmax(thetas[0]) * self.softmax(thetas[1]) + 
            (1 - self.softmax(thetas[0])) * (1 - self.softmax(thetas[1])))

    def NLL(self, params, model):
        if model == 'Early':
            c_a, c_v, sigma_a, sigma_v = params
            sigma_a = np.exp(sigma_a)
            sigma_v = np.exp(sigma_v)
            l_av = []
            l_a = np.prod([self.gauss_likelihood_a_v(j + 1, sigma_a, c_a, np.array(self.data)[0][j]) for j in range(0, 5)])
            l_v = np.prod([self.gauss_likelihood_a_v(j + 1, sigma_v, c_v, np.array(self.data)[1][j]) for j in range(0, 5)])
            
            for i in range(0, 5):
                for j in range(0, 5):
                    l_av.append(self.gauss_likelihood_av(j + 1, i + 1, sigma_a, sigma_v, c_a, c_v, np.array(self.data)[i+2][j]))
                    
            return -np.log(np.prod([l_a, l_v, np.prod(l_av)]))
    
        elif model == 'Fuzzy':
            l_av = []
            l_a = np.prod([self.binom_likelihood_a_v(params[j], np.array(self.data)[0][j]) for j in range(0, 5)])
            l_v = np.prod([self.binom_likelihood_a_v(params[j+5], np.array(self.data)[1][j]) for j in range(0, 5)])

            for i in range(0, 5):
                for j in range(0, 5):
                    l_av.append(self.binom_likelihood_av((params[i + 5], params[j]), np.array(self.data)[i+2][j]))
                    
            return -np.log(np.prod([l_a, l_v, np.prod(l_av)]))
        
        elif model == 'Late':
            c_a, c_v, sigma_a, sigma_v = params
            # Define mu
            mu_a = [1,2,3,4,5] # These are arbitrary 
            mu_v = [1,2,3,4,5] # These are arbitrary 

            # Define P i.e. the probability of a 'd' response given an auditory/visual/auditory+visual stimulus
            P_a  = stats.norm.cdf((mu_a - c_a)/sigma_a)
            P_v  = stats.norm.cdf((mu_v - c_v)/sigma_v)
            P_av = np.zeros((5,5))
            
            for i in range(5):
                for j in range(5):
                    P_av[i,j] = (P_a[j]*P_v[i])/(P_a[j]*P_v[i] + (1-P_a[j])*(1-P_v[i]))

            # Define the Negative Log Likelihood for the Binomial Distribution
            nll_combi_list = []
            for i in range(5):
                for j in range(5): 
                    nll_combi_list.append(np.log(stats.binom.pmf(self.x_av.iloc[i,j], self.n_responses, P_av[i,j])))
            #for i in range(0, 5):
            #   for j in range(0, 5):
            #      l_av.append(binom_likelihood_av((parameters[i + 5], parameters[j]), data[i+2][j]))
            nll = -sum(np.log(stats.binom.pmf(self.x_a, self.n_responses, P_a))) - \
                sum(np.log(stats.binom.pmf(self.x_v, self.n_responses, P_v))) - \
                sum(nll_combi_list)

            return nll

    def print_parameters(self, model):
        if model == 'Early':
            Initial_guess = [1,1,1,1]
            result = minimize(self.NLL, Initial_guess, model)
            print('Printing parameters for Early Fusion Model')
            print('c_a: ', result.x[0])
            print('c_v: ', result.x[1])
            print('sigma_a: ', np.exp(result.x[2]))
            print('sigma_v: ', np.exp(result.x[3]))
        elif model == 'Fuzzy':
            Initial_guess = np.random.uniform(size=10)
            result = minimize(self.NLL, Initial_guess, model)
            print('Printing parameters for Fuzzy Model')
            print("P_a1:", self.sig(result.x[0]))
            print("P_a2:", self.sig(result.x[1]))
            print("P_a3:", self.sig(result.x[2]))
            print("P_a4:", self.sig(result.x[3]))
            print("P_a5:", self.sig(result.x[4]))
            print("P_v1:", self.sig(result.x[5]))
            print("P_v2:", self.sig(result.x[6]))
            print("P_v3:", self.sig(result.x[7]))
            print("P_v4:", self.sig(result.x[8]))
            print("P_v5:", self.sig(result.x[9]))
        elif model == 'Late':
            Initial_guess = [0.5, 0.5, np.std(self.x_a), np.std(self.x_v)]
            result = minimize(self.NLL, Initial_guess, model)
            print('Printing parameters for Late Fusion Model')
            print('c_a: ', result.x[0])
            print('c_v: ', result.x[1])
            print('sigma_a: ', result.x[2])
            print('sigma_v: ', result.x[3])

        print("Negative Loglikelihood Value:", result.fun)

class BCI:
    '''
    BCI Class for parameter estimation and nll calculation

    input:
    n_responses: number of responses
    data: dataframe of the data
    experiment: 'audio' or 'visual': If audio then the experiment subject has to react on what they hear
    and if visual then the subject has to react on what they see. Will be relevant for the NLL calculation
    as it changes how to calculate the response probability of the audiovisual condition.

    parameters:
    sig(x): sigmoid function
    combined_distributions(means_atilde, means_vtilde, sigma_a_ini, sigma_v_ini): function to calculate 
    the combined distributions. Used by the NLL function
    NLL(params, model): Negative Log Likelihood function. Used for minimization
    print_parameters(model): prints the parameters of the model with the initialised data
    '''
    def __init__(self, n_responses, data, experiment='audio'):
        self.n_responses = n_responses
        self.data = data
        self.x_a = data.iloc[0,:]
        self.x_v = data.iloc[1,:]
        self.x_av = data.iloc[2:,:]
        self.experiment = experiment
        
    def sig(self, x):
        return 1/(1 + np.exp(-x))

    def combined_distributions(self, means_atilde, means_vtilde, sigma_a_ini, sigma_v_ini):
        w_a = sigma_v_ini**2 / (sigma_v_ini**2 + sigma_a_ini**2)
        # 5 by 5 matrix of 0 
        means_av_matrix= np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                mean_av = w_a * means_atilde[j] + (1 - w_a) * means_vtilde[i]
                means_av_matrix[i,j] = mean_av
        sigma_av = np.sqrt((sigma_a_ini**2 * sigma_v_ini**2) / (sigma_a_ini**2 + sigma_v_ini**2))
        return [means_av_matrix, sigma_av] 

    def NLL(self, params, model):
        if model == 'Early':
            c_a, c_v, sigma_a, sigma_v, c = params
            means = np.array([1, 2, 3, 4, 5])
            c = self.sig(c)
            means_atilde = means - c_a
            means_vtilde = means - c_v

            estimates = self.combined_distributions(means_atilde, means_vtilde, sigma_a, sigma_v)
            audiovisual_means = estimates[0]
            sigma_av = estimates[1]

            Pa = stats.norm.cdf((means - c_a)/sigma_a)
            Pv = stats.norm.cdf((means - c_v)/sigma_v)
            Pav = stats.norm.cdf((audiovisual_means)/sigma_av)
            if self.experiment == 'audio':
                P_av_BCI = c*Pav + (1-c)*np.array(Pa)
            elif self.experiment == 'visual':
                P_av_BCI = c*Pav + (1-c)*np.array(Pv)
        
        elif model == 'Strong Fusion':
            P_a1, P_a2, P_a3, P_a4, P_a5, P_v1, P_v2, P_v3, P_v4, P_v5, c = params
            P_a1, P_a2, P_a3, P_a4, P_a5 = self.sig(P_a1), self.sig(P_a2), self.sig(P_a3), self.sig(P_a4), self.sig(P_a5)
            P_v1, P_v2, P_v3, P_v4, P_v5 = self.sig(P_v1), self.sig(P_v2), self.sig(P_v3), self.sig(P_v4), self.sig(P_v5)
            c = self.sig(c)
            Pa  = [P_a1, P_a2, P_a3, P_a4, P_a5]
            Pv  = [P_v1, P_v2, P_v3, P_v4, P_v5]
            Pav = np.zeros((5,5)) # 5x5 matrix
            for i in range(5): # visual
                for j in range(5): # audio
                    Pav[i,j] = (Pa[j]*Pv[i])/(Pa[j]*Pv[i] + (1-Pa[j])*(1-Pv[i]))
            if self.experiment == 'audio':
                P_av_BCI = c*Pav + (1-c)*np.array(Pa)
            elif self.experiment == 'visual':
                P_av_BCI = c*Pav + (1-c)*np.array(Pv)


        elif model == 'Late':
            c_a, c_v, sigma_a, sigma_v, c = params
            c = self.sig(c)
            means = np.array([1, 2, 3, 4, 5])
            
            means_atilde = means - c_a
            means_vtilde = means - c_v

            Pa = stats.norm.cdf((means - c_a)/sigma_a)
            Pv = stats.norm.cdf((means - c_v)/sigma_v)
            Pav = np.zeros((5,5)) # 5x5 matrix
            for i in range(5): # visual
                for j in range(5): # audio
                    Pav[i,j] = (Pa[j]*Pv[i])/(Pa[j]*Pv[i] + (1-Pa[j])*(1-Pv[i]))  
            if self.experiment == 'audio':
                P_av_BCI = c*Pav + (1-c)*np.array(Pa)
            elif self.experiment == 'visual':
                P_av_BCI = c*Pav + (1-c)*np.array(Pv)

        L = []
        for i in range(5):
            for j in range(5):
                L.append(np.log(stats.binom.pmf(self.x_av.iloc[i,j], self.n_responses, P_av_BCI[i,j])))

        nll = -sum(np.log(stats.binom.pmf(self.x_a, self.n_responses, Pa))) - sum(np.log(stats.binom.pmf(self.x_v, self.n_responses, Pv))) - sum(L)
        return nll

    def print_parameters(self, model):
        if model == 'Early':
            Initial_guess = [0.5, 0.5, np.std(self.x_a), np.std(self.x_v), 0.5]
            result = minimize(self.NLL, Initial_guess, model)
            print('Printing parameters for Early Fusion Model')
            print('c_a: ', result.x[0])
            print('c_v: ', result.x[1])
            print('sigma_a: ', result.x[2])
            print('sigma_v: ', result.x[3])
            print('c: ', self.sig(result.x[4]))
        elif model == 'Strong Fusion':
            Initial_guess = [0.3] * 10 + [0]
            result = minimize(self.NLL, Initial_guess, model)
            print('Printing parameters for Strong Fusion Model')
            print("P_a1:", self.sig(result.x[0]))
            print("P_a2:", self.sig(result.x[1]))
            print("P_a3:", self.sig(result.x[2]))
            print("P_a4:", self.sig(result.x[3]))
            print("P_a5:", self.sig(result.x[4]))
            print("P_v1:", self.sig(result.x[5]))
            print("P_v2:", self.sig(result.x[6]))
            print("P_v3:", self.sig(result.x[7]))
            print("P_v4:", self.sig(result.x[8]))
            print("P_v5:", self.sig(result.x[9]))
            print("c:", self.sig(result.x[10]))
        elif model == 'Late':
            Initial_guess = [0.5, 0.5, np.std(self.x_a), np.std(self.x_v), 0.5]
            result = minimize(self.NLL, Initial_guess, model)
            print('Printing parameters for Late Fusion Model')
            print('c_a: ', result.x[0])
            print('c_v: ', result.x[1])
            print('sigma_a: ', result.x[2])
            print('sigma_v: ', result.x[3])
            print('c: ', self.sig(result.x[4]))

        print("Negative Loglikelihood Value:", result.fun)

class ROC:
    '''
    Roc class for calculating d_prime and plotting ROC curve

    input:
    TP: True Positive
    FP: False Positive
    TN: True Negative
    FN: False Negative

    parameters:
    d_prime: outputs the d_prime value
    roc_curve: plots the ROC curve with numbers of data points and ratings for the base and new stimuli images
    plot_roc_curve: plots the ROC curve from the given TP, FP, TN, FN values
    '''
    def __init__(self, TP, FP, TN, FN):
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN
        self.TPR = self.TP/(self.TP + self.FN)
        self.FPR = self.FP/(self.FP + self.TN)

    def d_prime(self, n_trials):
        print("d_prime for this model is: ", norm.ppf(self.TP/n_trials)-norm.ppf(self.FP/n_trials))

    def roc_curve(self, n_trials, n_datapoints, img_base_ratings, img_new_ratings):
        fpr = []
        tpr = []
        thresholds = np.linspace(0, n_trials, n_datapoints)
        for threshold in thresholds:

            #Stimuli
            yes_s = sum([1 if i >= threshold else 0 for i in img_new_ratings]) #tp 
            no_s = sum([1 if i < threshold else 0 for i in img_new_ratings]) #fn

            #No stimul
            yes_s0 = sum([1 if i >= threshold else 0 for i in img_base_ratings]) #fp
            no_s0 = sum([1 if i < threshold else 0 for i in img_base_ratings]) #tn

            tp = yes_s
            fn = no_s
            tn = no_s0
            fp = yes_s0
        
            fpr.append(fp / (fp + tn))
            tpr.append(tp / (tp + fn))

        plt.plot(fpr, tpr)
        plt.scatter(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    def plot_ROC(self, inverse=False):
        TPR = self.TPR
        FPR = self.FPR
        if inverse == True:
            TPR = norm.ppf(self.TPR)
            FPR = norm.ppf(self.FPR)
        plt.plot(FPR , TPR)
        plt.scatter(FPR, TPR)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    def print_parameters(self):
        print('True Positive Rate:', self.TPR)
        print('False Positive Rate:', self.FPR)
