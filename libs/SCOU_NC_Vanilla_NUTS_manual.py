import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

### NEW CIs
def get_95CI(signal, alpha = 5.0):
    CI95_lower = []
    CI95_upper = []

    for timestep in range(signal.shape[1]):

        # Gather the samples at time t:
        drawn_at_time_t = signal[:,timestep]

        # Computes the lower bound:
        lower_p = alpha / 2.0
        
        # Retrieves the observation at the lower percentile index:
        lower = np.percentile(drawn_at_time_t, lower_p)

        # Computes the upper bound:
        upper_p = (100 - alpha) + (alpha / 2.0)
        
        # Retrieves the observation at the upper percentile index:
        upper = np.percentile(drawn_at_time_t, upper_p)

        CI95_lower.append(lower)
        CI95_upper.append(upper)
        
    return np.array(CI95_lower), np.array(CI95_upper)
### END NEW CIs

# Define helper functions here
# Empirically much faster than numpy's, probably related to sample size I guess.
def normal_distribution_pdf(x, mean, std):
    
    A = 1 / (std * np.sqrt(2 * np.pi))
    B = - (1/2) * ((x - mean)/ std) ** 2
    
    return A * np.exp(B)

def approx_standard_normal_cdf_sw(x, loc=0, scale=1):
    '''
    Page's approximation of stdN cdf
    '''
    xx = (x - loc) / scale
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (xx + 0.044715 * xx**3)))

def censored_normal_logp(value, eps, latent):
    return pm.logcdf(pm.Normal.dist(mu=latent, sigma=eps), value)

# Fonction de densité pour la composante normale censurée
def censored_normal_logcdf(value, eps, latent):
    return pm.logcdf(pm.Normal.dist(mu=latent, sigma=eps), value)

# Fonction de densité pour la composante outlier censurée
def censored_uniform_logcdf(value, a, b):
    return pm.logcdf(pm.Uniform.dist(lower=a, upper=b), value)

# Fonction de densité pour la composante outlier censurée
def censored_uniform_logp(value, a, b):
    return pm.logcdf(pm.Uniform.dist(lower=a, upper=b), value)

class SCOU_RW1_NUTS():

    def __init__(self, observations,
                 censoring_threshold_lod_vect=1,
                 p_out_frozen=-1,
                 tuning_iters=4000,
                 sampling_iters=2000,
                 nb_chains=1,
                 export_name='default.nc',
                 export_chains=False,
                 RW_order=1):

        self.observations = observations
        self.censoring_threshold_lod_vect = censoring_threshold_lod_vect
        self.nsteps = self.observations.shape[0]
        self.rng = np.random.default_rng(666)
        self.tuning_iters = tuning_iters
        self.sampling_iters = sampling_iters
        self.nb_chains = nb_chains
        self.export_name = export_name
        self.export_chains = export_chains
        self.p_out_frozen = p_out_frozen
        self.RW_order = RW_order


    def obs_discrimination(self):

        self.unobserved_indexes = np.where(np.isnan(self.observations))[0]
        self.observations_below_LoD = np.where(self.observations<=self.censoring_threshold_lod_vect)[0]
        self.observations_above_LoD = np.where(self.observations>self.censoring_threshold_lod_vect)[0]
        self.T_ronde = np.setdiff1d(np.arange(self.observations.shape[0]), self.unobserved_indexes)
        self.borne_inf, self.borne_sup = np.nanmin(self.observations) - 2*np.nanstd(self.observations), np.nanmax(self.observations) + 2*np.nanstd(self.observations)


    def model_definition(self):

        self.obs_discrimination()

        with pm.Model() as self.all_processes_1:

            # Priors pour les paramètres
            sig = pm.InverseGamma('sig', alpha=2, beta=1)
            eps = pm.InverseGamma('eps', alpha=2, beta=1)
            if self.p_out_frozen==-1:
                p_out = pm.Beta('p_out', alpha=2, beta=5)
            else:
                p_out = self.p_out_frozen

            if self.RW_order==1:
                # Processus latent X[t] (AR(1))
                latent = pm.AR("latent", rho=np.array([1]), sigma=sig, shape=self.nsteps)
            elif self.RW_order==2:
                # Processus latent X[t] (AR(2))
                latent = pm.AR("latent", rho=np.array([2, -1]), sigma=sig, shape=self.nsteps)

            # Composante gaussienne pour les données non censurées
            normal_component = pm.Normal.dist(mu=latent[self.observations_above_LoD], sigma=eps)
            
            # Composante pour les outliers (distribution uniforme)
            outlier_component = pm.Uniform.dist(lower=self.borne_inf, upper=self.borne_sup)

            # Modèle de mélange pour les données non censurées
            obs_uncensored = pm.Mixture(
                'obs_uncensored',
                w=[1 - p_out, p_out],
                comp_dists=[normal_component, outlier_component],
                observed=self.observations[self.observations_above_LoD]
            )

            # Données censurées:

            normal_censored_component = pm.DensityDist.dist(
                eps, latent[self.observations_below_LoD],
                logp=censored_normal_logp,
                logcdf=censored_normal_logcdf,
                class_name="normal_censored_component",
            )

            outlier_censored_component = pm.DensityDist.dist(
                self.borne_inf, self.borne_sup, 
                logp=censored_uniform_logp,
                logcdf=censored_uniform_logcdf,
                class_name="outlier_censored_component",
            )

            # Mélange pour les données censurées
            obs_censored = pm.Mixture(
                'obs_censored',
                w=[1 - p_out, p_out],
                comp_dists=[normal_censored_component, outlier_censored_component],
                observed=self.observations[self.observations_below_LoD]
            )

    def fit(self):

        self.model_definition()

        # Inférence
        with self.all_processes_1:
            self.trace_all_1 = pm.sample(self.sampling_iters, tune=self.tuning_iters, 
                                    chains=self.nb_chains, 
                                    return_inferencedata=True, 
                                    random_seed=self.rng)

        self.params = ['sig', 'eps'] 
        if self.p_out_frozen==-1:
            self.params = ['sig', 'eps', 'p_out'] 

        print("Raw summary:")
        print(az.summary(self.trace_all_1, var_names=self.params))
        self.params_summary = az.summary(self.trace_all_1, var_names=self.params)


        if self.export_chains:
            self.trace_all_1.to_netcdf(self.export_name)


    def predict(self, selected_chains):

        self.latent_posterior_distribution = self.trace_all_1['posterior']['latent'].values[selected_chains].reshape(len(selected_chains)*self.sampling_iters, -1)
        self.muX = self.latent_posterior_distribution.mean(axis=0)
        self.CIL, self.CIU = get_95CI(self.latent_posterior_distribution)
        self.compute_pointwise_outlier_probabilities(selected_chains)

        print("Best chain combination summary:")
        print(az.summary(self.trace_all_1.sel(chain=selected_chains), var_names=self.params))


    def compute_pointwise_outlier_probabilities(self, selected_chains):

        nb_draws = self.trace_all_1['posterior']['eps'].values[selected_chains].shape[1] * len(selected_chains)

        self.pointwise_pout = np.ones(self.observations.shape[0]) * np.nan
        self.pointwise_pout_dist = np.ones((self.observations.shape[0], nb_draws)) * np.nan

        # Vectorizing these computations first so that we don't have to repeat them in the next for loop:
        this_partial_emission_vector = np.ones(self.observations.shape[0]) * (1/(self.borne_sup - self.borne_inf))
        this_partial_emission_vector[self.observations_below_LoD] = ((self.censoring_threshold_lod_vect[self.observations_below_LoD] - self.borne_inf)/(self.borne_sup - self.borne_inf))

        for this_timestep in self.T_ronde:     
            xhat_t = self.observations[this_timestep]
            x_t = self.trace_all_1['posterior']['latent'].values[selected_chains].reshape(len(selected_chains)*self.sampling_iters, -1)[:, this_timestep]
            this_epsilon = self.trace_all_1['posterior']['eps'].values[selected_chains].reshape(len(selected_chains)*self.sampling_iters, )[:,]

            if self.p_out_frozen==-1:
                this_pout = self.trace_all_1['posterior']['p_out'].values[selected_chains].reshape(len(selected_chains)*self.sampling_iters, )[:,]
            else:
                this_pout = self.p_out_frozen

            if this_timestep in self.observations_below_LoD:
                num = this_pout * this_partial_emission_vector[this_timestep]
                denom_not_outlier = (1-this_pout) * approx_standard_normal_cdf_sw(xhat_t, x_t, this_epsilon)
                denom = denom_not_outlier + num
            
            elif this_timestep in self.observations_above_LoD:    
                num = this_pout * this_partial_emission_vector[this_timestep]
                denom_not_outlier = (1-this_pout) * normal_distribution_pdf(xhat_t, x_t, this_epsilon)
                denom = denom_not_outlier + num

            num = np.array(num)
            denom = np.array(denom)
            
            self.pointwise_pout_dist[this_timestep] = (num/denom)
            self.pointwise_pout[this_timestep] = np.mean(self.pointwise_pout_dist[this_timestep])

    def visualize_latents(self, selected_chains):
        plt.figure()
        for i in range(self.nb_chains):
            plt.plot(self.trace_all_1['posterior']['latent'][i].mean(axis=0), label=i)

        plt.title('Raw chains')
        plt.legend()
        plt.show()

        plt.figure()
        for i in selected_chains:
            plt.plot(self.trace_all_1['posterior']['latent'][i].mean(axis=0), label=i)

        plt.title('Optimized chains')
        plt.legend()
        plt.show()

            

    



