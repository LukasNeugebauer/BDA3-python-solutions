
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

# custom cauchy pdf
def cauchypdf(y,theta):
    return 1/(1+(y-theta)**2)

# likelihood
def wrap_likelihood(y):
    def likelihood(theta):
        ll = np.prod([cauchypdf(x, theta) for x in y])
        return ll
    return likelihood

# very important function
def prior():
    return 1/100

# !!! the values for y are different than the ones given in the exercise. They match those in the solution !!!
y = np.r_[-2, -1, 0, 1.5, 2.5]
likelihood = wrap_likelihood(y)

# very important function
def posterior(theta):
    return prior() * likelihood(theta)

m = 1000
# !!! the values for theta are different than the ones given in the exercise. They match those in the solution !!!
theta = np.linspace(0, 1, m)
post = np.array([posterior(t) for t in theta])
post /= post.sum() / m # normalize the posterior
sample = np.random.choice(theta, p=post / post.sum(), size=1000)
pred_samples = np.array([cauchy(loc=loc, scale=1).rvs() for loc in sample])

plt.figure()
plt.plot(post)
plt.ylim([0, max(post)*1.1])
plt.xlabel('Theta'); plt.ylabel('Normalized Density')

plt.figure()
plt.hist(sample)
plt.xlabel('Theta');

# the cauchy pdf often gets WILD results. better put some constraints on the histogram
plt.figure()
plt.hist(pred_samples,
         bins=np.linspace(np.percentile(pred_samples, 5),
                          np.percentile(pred_samples, 95), 100))
plt.xlabel('New Observation')