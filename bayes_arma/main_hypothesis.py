import numpy as np
from scipy.linalg import hankel
from scipy.stats import multivariate_normal
from bayes_arma.util import generate_ar
from scipy.special import logsumexp
np.random.seed(123)
print('\n')


def factor_gaussian_prob(mean, var, variance):
    return -(len(mean)) / 2 * np.log(2 * np.pi * variance) - 1 / (2 * variance) * np.sum(np.square(mean - var))


def generate_evidence_samples(num_samples, noise_var):
    for num_sample in range(num_samples):
        alphas = prior.rvs()
        yield factor_gaussian_prob(y_t_out, H.dot(alphas), noise_var)


# True AR(p) coefficents
ar_coeff = [0.8, -0.2, 0.3]

# Modelling assumptions
prior_var = 1.0
noise_var = 0.25

# Generate some data
N = 100  # number of time steps
y_t = generate_ar(ar_coeff, noise_var, N)

log_evidences = []
for p in range(1, 7):
    prior = multivariate_normal(cov=prior_var * np.eye(p))
    num_samples = 1000

    # Prepare data for efficient calculation
    H = hankel(y_t[:-p], y_t[-(p+1):-1])
    y_t_out = y_t[p:]

    # Do posterior inference
    post_cov = noise_var * np.linalg.inv(H.T.dot(H) + (noise_var / prior_var) * np.eye(p))
    post_mean = post_cov.dot(H.T).dot(y_t_out) / noise_var
    log_likelihood = factor_gaussian_prob(y_t_out, H.dot(post_mean), noise_var)

    # First approximate with MC
    log_evidence = logsumexp(list(generate_evidence_samples(num_samples, noise_var))) - np.log(num_samples)
    log_evidences.append(log_evidence)

    # Second approximate with BIC
    BIC = - 1/2 * np.log(N) * p
    log_evidence_BIC = log_likelihood + BIC

    # Third approximate with Laplace
    log_p_post_mean = factor_gaussian_prob(post_mean, np.zeros_like(post_mean), prior_var)
    occam_factor = log_p_post_mean + np.log(np.linalg.det(np.pi * post_cov))
    log_evidence_occam = log_likelihood + occam_factor

    print(f'  at AR({p:.0f}) MC approximation {10 * log_evidence/np.log(10):12.3f} dB -- '
          f'BIC approximation {10* log_evidence_BIC / np.log(10):12.3f} dB -- '
          f'Laplace approximation {10 * log_evidence_occam / np.log(10):12.3f} dB')

log_evidences_sorted = np.sort(log_evidences)[::-1]
factor = log_evidences_sorted[0] - log_evidences_sorted[1]
print('\n'*5)
print(f'With MC approximation, the best hypothesis is {np.exp(factor)} better than the second best')
