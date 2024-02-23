from numpy.random import RandomState
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from torch import nn
from torch import optim
import torch.distributions as D

norm_multinom = create_fun_with_mem()

def _pc_multinomial(N, K):
    """parametric complexity for multinomial distributions.

    Args:
        N (int): number of data.
        K (int): number of clusters.

    Returns:
        float: parametric complexity for multinomial distributions.
    """
    return norm_multinom.evaluate(N, K)

def _log_pc_gaussian(N_list, D, R, lmd_min):
    """log parametric complexity for a Gaussian distribution.

    Args:
        N_list (np.ndarray): list of the number of data.
        D (int): dimension of data.
        R (float): upper bound of ||mean||^2.
        lmd_min (float): lower bound of the eigenvalues of the covariance
            matrix.

    Returns:
        np.ndarray: list of the parametric complexity.
    """
    N_list = np.array(N_list)

    log_PC_list = sum([
        D * N_list * np.log(N_list / 2 / math.e) / 2,
        (-1) * D * (D - 1) * np.log(math.pi) / 4,
        (-1) * np.sum(
            loggamma((N_list.reshape(-1, 1) - np.arange(1, D + 1)) / 2),
            axis=1
        ),
        (D + 1) * np.log(2 / D),
        (-1) * loggamma(D / 2),
        D * np.log(R) / 2,
        (-1) * D**2 * np.log(lmd_min) / 2
    ])

    return log_PC_list


def log_pc_gmm(K_max, N_max, D, *, R=1e+3, lmd_min=1e-3):
    """log PC of GMM.

    Calculate (log) parametric complexity of Gaussian mixture model.

    Args:
        K_max (int): max number of clusters.
        N_max (int): max number of data.
        D (int): dimension of data.
        R (float): upper bound of ||mean||^2.
        lmd_min (float): lower bound of the eigenvalues of the covariance
            matrix.

    Returns:
        np.ndarray: array of (log) parametric complexity.
            returns[K, N] = log C(K, N)
    """
    log_PC_array = np.zeros([K_max + 1, N_max + 1])
    r1_min = D + 1

    # N = 0
    log_PC_array[:, 0] = -np.inf

    # K = 0
    log_PC_array[0, :] = -np.inf

    # K = 1
    # N <= r1_min
    log_PC_array[1, :r1_min] = -np.inf
    # N > r1_min
    N_list = np.arange(r1_min, N_max + 1)
    log_PC_array[1, r1_min:] = _log_pc_gaussian(
        N_list,
        D=D,
        R=R,
        lmd_min=lmd_min
    )

    # K > 1
    for k in range(2, K_max + 1):
        for n in range(1, N_max + 1):
            r1 = np.arange(n + 1)
            r2 = n - r1
            log_PC_array[k, n] = logsumexp(sum([
                loggamma(n + 1),
                (-1) * loggamma(r1 + 1),
                (-1) * loggamma(r2 + 1),
                r1 * np.log(r1 / n + 1e-100),
                r2 * np.log(r2 / n + 1e-100),
                log_PC_array[1, r1],
                log_PC_array[k - 1, r2]
            ]))

    return log_PC_array

def _comp_loglike(*, X, Z, rho, means, covariances):
    """complete log-likelihood

    Args:
        X (ndarray): Data (shape = (N, K)).
        Z (ndarray): Latent variables (shape = (N,)).
        rho (ndarray): Mixture proportion (shape = (K,)).
        means (ndarray): Mean vectors (shape = (K, D)).
        covariances (ndarray): Covariance matrices (shape = (K, D, D)).
    Returns:
        float: Complete log likelihood.
    """
    _, D = X.shape
    K = len(means)
    nk = np.bincount(Z, minlength=K)

    if min(nk) <= 0:
        return np.nan
    else:
        c_loglike = 0
        for k in range(K):
            #print(f'current c_loglike = {c_loglike} with k = {k} out of {range(K)}...')
            c_loglike += nk[k] * np.log(rho[k])
            #print(f'nk[k]={nk[k]}; rho[k]={rho[k]}; c_loglike={c_loglike}')
            c_loglike -= 0.5 * nk[k] * D * np.log(2 * math.pi * math.e)
            #print(f'np.log(2 * math.pi * math.e)={np.log(2 * math.pi * math.e)}; c_loglike={c_loglike}')
            c_loglike -= 0.5 * nk[k] * np.log(np.linalg.det(covariances[k]))
            #print(f'covariances[k]={covariances[k]};np.log(np.linalg.det(covariances[k])={np.log(np.linalg.det(covariances[k]))}; c_loglike={c_loglike}')
        return c_loglike

from numpy.random import RandomState
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

class GMMUtils():
    """Useful Functions for Gaussian Mixture Model.
    """

    def __init__(self, rho, means, covariances):
        """
        Args:
            rho (ndarray): Mixture proportion (shape = (K,)).
            means (ndarray): Mean vectors (shape = (K, D)).
            covariances (ndarray): Covariance matrices (shape = (K, D, D)).
        """
        self.rho = rho
        self.means = means
        self.covariances = covariances
        self.K = len(rho)

    def sample(self, N=100, random_state=None):
        """Sample from GMM.

        Args:
            N (int): Number of the points to sample.
            random_state (Optional[int]): Random state.
        Returns:
            ndarray: Sampled points (shape = (N, K)).
        """
        random = RandomState(random_state)
        nk = random.multinomial(N, self.rho)
        X = []
        for mean, cov, size in zip(self.means, self.covariances, nk):
            X_new = multivariate_normal.rvs(
                mean=mean,
                cov=cov,
                size=size,
                random_state=random
            )
            if size == 0:
                pass
            elif size == 1:
                X.append(X_new)
            else:
                X.extend(X_new)
        return np.array(X)

    def logpdf(self, X):
        """Calculate log pdf.

        Args:
            X (ndarray): Data (shape = (N, D)).
        Returns:
            ndarray: Matrix of log pdf (shape = (N, K)).
        """
        N = len(X)
        log_pdf = np.zeros([N, self.K])
        for k in range(self.K):
            log_pdf[:, k] = multivariate_normal.logpdf(
                X,
                self.means[k],
                self.covariances[k],
                allow_singular=True
            )
        return log_pdf

    def prob_latent(self, X):
        """Probability of the latent variables.

        Args:
            X (ndarray): Data (shape = (N, D)).
        Returns:
            ndarray: Matrix of latent probabilities (shape = (N, K)).
        """
        log_pdf = self.logpdf(X)
        log_rho_pdf = np.log(self.rho + 1e-50) + log_pdf
        log_prob = (
            log_rho_pdf -
            logsumexp(log_rho_pdf, axis=1).reshape((-1, 1))
        )
        return np.exp(log_prob)

import torch
from torch import nn
from torch import optim
import torch.distributions as D

class GaussianMixtureModel(torch.nn.Module):
    # https://discuss.pytorch.org/t/fit-gaussian-mixture-model/121826

    def __init__(self, n_components: int=7):
        super().__init__()
        weights = torch.ones(n_components, )
        means   = torch.randn(n_components, )
        stdevs  = torch.tensor(np.abs(np.random.randn(n_components, )))
        self.weights = torch.nn.Parameter(weights)
        self.means   = torch.nn.Parameter(means)
        self.stdevs  = torch.nn.Parameter(stdevs)
        
    def _update_parameters(self, new_weights, new_means, new_stdevs):
        self.weights = torch.nn.Parameter(new_weights)
        self.means   = torch.nn.Parameter(new_means)
        self.stdevs  = torch.nn.Parameter(new_stdevs)
        
    def _get_weights(self):
        return self.weights
    
    def _get_parameters(self):
        return self.weights, self.means, self.stdevs
    
    def forward(self, x):
        #print(self.weights)
        mix  = D.Categorical(self.weights)
        #std_weight = 1e-4
        #comp = D.Normal(self.means, std_weight * self.stdevs.abs())
        comp = D.Normal(self.means, self.stdevs)
        gmm  = D.MixtureSameFamily(mix, comp)
        return - gmm.log_prob(x).mean()



class GMMModelSelection():
    """Model Selection of Gaussian Mixture Model.
    """
    def __init__(
        self,
        K_min=1,
        K_max=20,
        reg_covar=1e-3,
        random_state=None,
        mode='GMM_BIC',
        weight_concentration_prior=1.0,
        tol=1e-3,
        degrees_of_freedom_prior=None,
    ):
        """
        Args:
            K_max (int): Maximum number of the components.
            reg_covar (float): Reguralization for covariance.
            random_state (Optional[int]): Random state.
            mode (str): Estimation mode. Choose from the following:
                'GMM_BIC' (EM algorithm + BIC)
                'GMM_DNML' (EM algorithm + DNML)
                'BGMM' (Variational Bayes based on Dirichlet distribution).
            weight_concentration_prior (float): Weight concentration prior
                for BGMM.
            tol (float): Tolerance for GMM convergence.
        """
        self.K_max = K_max
        self.K_min = K_min
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.mode = mode
        self.weight_concentration_prior = weight_concentration_prior
        self.tol = tol
        self.degrees_of_freedom_prior = degrees_of_freedom_prior

    def fit(self, X):
        """Select the best model.

        Args:
            X (ndarray): Data (shape = (N, K)).
        """
        
        if self.mode == 'GMM_DNML':
            log_pc_array = log_pc_gmm(
                K_max=self.K_max,
                N_max=X.shape[0],
                D=X.shape[1]
            )

        if self.mode in ['GMM_BIC', 'GMM_DNML']:
            model_list = []
            criterion_list = []
            for K in range(self.K_min, self.K_max + 1):
                #print(f'Fitting a GMM with K={K} components...')
                # Fit
                model_new = GaussianMixture(
                    n_components=K,
                    reg_covar=self.reg_covar,
                    random_state=self.random_state,
                    n_init=10,
                    tol=self.tol,
                    max_iter=10000
                )
                model_new.fit(X)
                model_list.append(model_new)
                # Calculate information criterion
                if self.mode == 'GMM_BIC':
                    criterion_list.append(model_new.bic(X))
                elif self.mode == 'GMM_DNML':
                    Z = model_new.predict(X)
                    loglike = _comp_loglike(
                        X=X,
                        Z=Z,
                        rho=model_new.weights_,
                        means=model_new.means_,
                        covariances=model_new.covariances_
                    )
                    complexity = np.log(_pc_multinomial(len(X), K))
                    for k in range(K):
                        Z_k = sum(Z == k)
                        if log_pc_array[1, Z_k] != - np.inf:
                            complexity += log_pc_array[1, Z_k]
                    criterion_list.append(- loglike + complexity)
                    #print(f'loglike:{loglike}, complexity:{complexity}')
                    #print(criterion_list)
            idx_best = np.nanargmin(criterion_list)
            self.model_best_ = model_list[idx_best]

        elif self.mode == 'BGMM':
            self.model_best_ = BayesianGaussianMixture(
                n_components=self.K_max,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                weight_concentration_prior=self.weight_concentration_prior,
                weight_concentration_prior_type='dirichlet_distribution',
                max_iter=10000,
                n_init=10,
                tol=self.tol,
                degrees_of_freedom_prior=self.degrees_of_freedom_prior
            )
            self.model_best_.fit(X)
        else:
            raise ValueError('methods should be GMM_BIC, GMM_DNML or BGMM.')

        self.K_ = self.model_best_.n_components
        self.rho_ = self.model_best_.weights_
        self.means_ = self.model_best_.means_
        self.covariances_ = self.model_best_.covariances_
        return(criterion_list)

    def prob_latent(self, X):
        """Probability of the latent variables.

        Args:
            X (ndarray): Data (shape = (N, D)).
        Returns:
            ndarray: Matrix of latent probabilities (shape = (N, K)).
        """
        analysis = GMMUtils(
            rho=self.rho_,
            means=self.means_,
            covariances=self.covariances_
        )
        return analysis.prob_latent(X)

    def predict(self, X):
        """Predict latent labels.

        Args:
            X (ndarray): Data (shape = (N, D)).
        Returns:
            ndarray: predicted labels (shape = (N,)).
        """
        prob_latent_ = self.prob_latent(X)
        return np.argmax(prob_latent_, axis=1)