import jax  # Automatic differentiation and accelerated array operations (needed for gradient calculation)
import jax.numpy as jnp # it is like numpy but with added capabilities like differentiability and support for GPU acceleration
import numpyro # probabilistic programming tools for Bayesian inference
from numpyro.infer import MCMC, NUTS # MCMC executes HMC algorithm, NUTS : improved variant of HMC (automatically tunes the number of leapfrog steps and step sizes)
import numpyro.distributions as dist

# Potential energy function : - the log of the target distribution
def potential_energy(q):
    return 0.5 * jnp.sum(q ** 2)
#(Does jax take as input the -log or the distribution it self)?

def model():
    x = numpyro.sample('x', dist.Normal(0, 1))  # Sample the position x
    V_x = potential_energy(x)
    numpyro.factor('potential_energy', -V_x)  # Use numpyro.factor to guide HMC to favor values of x that minimize the potential energy.
# the model function implicitly returns a distribution from which HMC or NUTS will sample.

nuts_kernel = NUTS(model) # automatically adapts the step size and the number of leapfrog steps

mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000) # This is the actual sampling process : num_warmup = number of burn-in steps,
                                                           # num_samples = the number of samples you want to collect from the target distribution after the warmup
# the output should be an array of num_samples samples

q0 = jnp.zeros((1,)) # intialise the position

mcmc.run(jax.random.PRNGKey(0)) # Run the sampler (beginning exploration fron q0)
# jax.random.PRNGKey(0): JAX uses a special way of handling random numbers. This line sets the random seed (using 0 here)

samples = mcmc.get_samples() ['x'] # retreiving samples (samples of x) : an array

#print(samples)
print(f'<x> = {jnp.mean(samples)}')
