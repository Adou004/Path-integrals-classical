# MCMC
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


number_of_paths = 10000 # Pour mini choisir number_of_paths un multiple de jump (number_of_paths = integer * jump)
# for 𝛿t in range (0.1, 1.2, 0.2):
𝛿t = 0.1
N𝜏 = int(250 / 𝛿t)
# N𝜏 * 𝛿t = 250 ; total time of the latice is 250 (seconds for example)
m = 𝜔 = 𝛿t
h = 5 # [-h, h] is the interval fram which the change in position is proposed
jump = 100
M_X2 = []
PATHS = []
H = []
x = np.zeros(N𝜏)

def Jackknife(samples_list:list, batch_size:int):
    batch_size = int(batch_size)
    if len(samples_list) % batch_size != 0:
        samples_list = samples_list[len(samples_list) % batch_size :]
    num_batches = len(samples_list)//batch_size
    estimator= np.zeros(num_batches)
    for i in range (num_batches):
        estimator[i]= ( np.sum(samples_list) - np.sum(samples_list[ i*batch_size : (i+1)*batch_size ]) ) / (len(samples_list)-batch_size)
    variance = 0
    for i in range (num_batches):
        variance += (estimator[i]- np.mean(samples_list))**2
    variance = variance * (num_batches-1) / num_batches
    error = np.sqrt(variance)
    return error

"""
def TPACF(samples:list): # Two-point autocorrelation function
    N = len(samples)
    NACF_list = []
    for tmc in range(N): 
        Ao_list = [( samples [i]- np. mean (samples [:N-tmc]) )*( samples [i+tmc]-np. mean (samples [tmc:]) ) for i in range (N-tmc)]
        NACF_list.append(np.mean(Ao_list))
    # Ao_0 <==> tmc = 0
    Ao_0 = np.mean([( samples[i]- np. mean (samples) )**2 for i in range (N)]) # THIS LOOKS LIKE STANDARD DIVIATION
    NACF_list = NACF_list / Ao_0
    return NACF_list


def Tau_exp (ACF:list):
    def exponential_decay(t, A_0, tau):
        return A_0 * np.exp(- t / tau)

    xdata = np.arange(len(ACF))
    ydata = ACF

    # Perform curve fitting
    popt, pcov = curve_fit(exponential_decay, xdata, ydata)

    # Optimal parameters
    A_0, tau = popt
    print (f'tau_exp = {tau}')

    # Generate model predictions using the optimal parameters
    yfit = exponential_decay(xdata, A_0, tau)
    # Plot the original data and the fitted curve
    plt.scatter(xdata, ydata, label='Data')
    plt.plot(xdata, yfit, color='red')
    plt.legend()
    plt.show()
    
    return tau
"""

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]
# you are interested in computing the autocorrelation for all possible lags (including negative lags) and then selecting the relevant portion (non-negative lags). Using full allows you to compute the complete correlation before discarding the negative lags. If you were to use mode='same', you would lose the information about how the signal correlates at longer lags, which is often crucial in statistical mechanics and time series analysis, particularly in contexts like Monte Carlo simulations or quantum computing applications.

def integrated_correlation_time(x):
    autocorrelation = autocorr(x)
    return 0.5 + np.sum(autocorrelation[1:] / autocorrelation[0])

def G2_tau (PATHS:list): # Two-point autocorrelation function
    N = len(PATHS)
    G2_delta_tau_list = []
    for delta_tau in range(N):
        G2_delta_tau_list.append(np.mean( [ PATHS[tau] * PATHS[tau+delta_tau] for tau in range(N-delta_tau)] ))
    Log_G2_delta_tau_list= np.log(G2_delta_tau_list)
    X = np.arange(N)
    plt.plot(X, Log_G2_delta_tau_list)
    plt.show()




for j in range (number_of_paths + 1):

    accrate = 0.0 # initialise the Acceptace rate
    Tar_accrate = 0.5  # Target acceptance rate

    for τ in range (N𝜏):
            
        τmin = (τ + Nτ - 1) % Nτ  # Previous time slice, with periodic boundary condition
        τplu = (τ + 1) % Nτ       # Next time slice, with periodic boundary condition

        x_new = x[τ] + 2*h * (np.random.rand() - 0.5) # The proposed change
                
        S = m/2 * ( (x[τplu] - x[τ])**2 + (x[τ] - x[τmin])**2 + 𝜔**2 * x[τ]**2 ) # The actual action ( at x[τ] )
        S_new = m/2 * ( (x[τplu] - x_new)**2 + (x_new - x[τmin])**2 + 𝜔**2 * x_new**2 ) # The new action ( at x_new )
        
        if np.random.rand() < np.exp(- (S_new - S)):  # accept the new position with probability min (1, exp(-𝛥S))
            x[τ] = x_new
            accrate += 1.0 / Nτ 
    
    h = h * accrate / Tar_accrate # modify the radius of the interval [-h, h] according to the acceptance rate
    

    
    PATHS.extend(x * 𝛿t)               #  Every "jump" paths, the current path is stored in PATHS
    M_X2.append(np.mean(x**2) * 𝛿t**2) #  Every "jump" paths, the average value of the (position squared) is stored in M_X2
        ## Knowing that E_0 = 𝛿t^2 * < ( x{tilde} )^2 >  ; in other words E_0 = <x^2>
    H.append(h)

print(f'<x> = {np.mean(PATHS)} ~ 0')
print(f'E_0 = {np.mean(M_X2)} ~ 0.5')
#print(f'h = {np.mean(H)}')
print(f'integrated_correlation_time : {integrated_correlation_time(M_X2)}')

#G2_tau(PATHS)

X = np.arange(-3,3,0.1)
Y = ((np.pi * m*𝜔/𝛿t**2)**(-1/4)* np.exp(-X**2 *m*𝜔/𝛿t**2/2))**2
plt.hist(PATHS, bins=1000, density=True)
plt.plot(X, Y, 'r')
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$|\psi_0|^2$', fontsize=20)
plt.show()









"""
tau_exp = int( Tau_exp(TPACF(M_X2)) )
tau_o = 2 * ( np.sum( TPACF(M_X2) [1 : 3 * int(tau_exp)] ) ) + 1    #  3 * int(tau_exp)
print (f'Autocorrelation time of <x^2> is {tau_o} paths')

xpts = np.arange(0, len(M_X2))
plt.plot(xpts, M_X2, linestyle='-', marker='o', label=f'delta_t = {𝛿t}')
plt.plot(xpts, np.ones(len(xpts)) * np.mean(M_X2), 'r')
plt.plot(xpts, np.ones(len(xpts)) * 0.5          , 'g')
#plt.plot(xpts, H, 'g', label=f'delta_t = {𝛿t}')
plt.legend()
plt.show()

std = np.std(M_X2, ddof=1)/np.sqrt(len(M_X2))
jack = Jackknife(M_X2, tau_o+1)
print(f'jack = {jack}', f'standard div = {std}')


# Uncorrelated 

paths = []
m_x2 = []
for j in range(len(PATHS)):
    if j % tau_o == 0:
        paths.append(PATHS[j])
        m_x2.append(M_X2[j])
print(f'<x> = {np.mean(paths)} ~ 0')
print(f'E_0 = {np.mean(m_x2)} ~ 0.5')       


#plt.boxplot()

X = np.arange(-3,3,0.1)
Y = ((np.pi * m*𝜔/𝛿t**2)**(-1/4)* np.exp(-X**2 *m*𝜔/𝛿t**2 /2))**2
plt.hist(paths, bins=1000, density=True)
plt.plot(X, Y, 'r')
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$|\psi_0|^2$', fontsize=20)
plt.show()

X = np.arange(-3,3,0.1)
Y = ((np.pi * m*𝜔/𝛿t**2)**(-1/4)* np.exp(-X**2 *m*𝜔/𝛿t**2/2))**2
plt.hist(PATHS, bins=1000, density=True)
plt.plot(X, Y, 'r')
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$|\psi_0|^2$', fontsize=20)
plt.show()

xpts = np.arange(tau_o, number_of_paths+1, tau_o)
plt.plot(xpts, m_x2, linestyle='-', marker='o', label=f'delta_t = {𝛿t}')
plt.plot(xpts, np.ones(len(xpts)) * np.mean(M_X2), 'r')
plt.plot(xpts, np.ones(len(xpts)) * np.mean(m_x2), 'b')
plt.plot(xpts, np.ones(len(xpts)) * 0.5          , 'g')
#plt.plot(xpts, H, 'g', label=f'delta_t = {𝛿t}')
plt.legend()
plt.show()
"""
