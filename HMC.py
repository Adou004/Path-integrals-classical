import jax
import jax.numpy as jnp
from jax import grad
import numpyro as pyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt

𝛿t = 0.5
# System parameters
m = omega = 𝛿t
n_positions = int(250 / 𝛿t)
h = 100 # h should be big h>>0.5 because HMC approaches the mean of q 'zero'
num_samples = 10000

# Potential energy function (action)
def potential_energy(q):
    kinetic_term =  jnp.sum (m/2 * ( jnp.diff(q, prepend=q[-1])**2 ))
    potential_term = jnp.sum ((m/ 2 * omega**2) * q**2)
    return kinetic_term + potential_term

# Define the NumPyro model
def model(): 
    q = pyro.sample('q', dist.Uniform(low=-h, high=h).expand([n_positions]))
    pyro.factor("potential_energy", - potential_energy(q))

# Run HMC sampling with NUTS for automatic tuning
nuts_kernel = pyro.infer.NUTS(model)  # Automatically adapts step size and leapfrog steps
mcmc = pyro.infer.MCMC(nuts_kernel, num_warmup=500, num_samples= num_samples)
mcmc.run(jax.random.PRNGKey(1))  # Different key for potentially different samples
#mcmc.print_summary()

# Get the samples
samples = mcmc.get_samples()
q_samples = samples['q'] * 𝛿t

# Print results
print(f"Mean of q: {jnp.mean(q_samples)} ~ 0.0 ")
print(f"Mean of q^2: {jnp.mean(q_samples**2)} ~ 0.5 ")

Q = q_samples.flatten()

X = jnp.arange(-3,3,0.05)
Y = ((jnp.pi * m*omega/𝛿t**2)**(-1/4)* jnp.exp(-X**2 *m*omega/𝛿t**2/2))**2
plt.hist(Q, bins=1000, density=True)
plt.plot(X, Y, 'r')
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$|\psi_0|^2$', fontsize=20)
plt.show()

