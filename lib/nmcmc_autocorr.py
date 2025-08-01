#Modified version of what they used in HAN codebase.
import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt


def trans_conf(config,n_y,L): ## shift configuration of spins in y direction by n_y=0,1,..,L-1
    
    new_config = torch.cat( (config[n_y:L], config[:n_y]) , dim=0 ) 
    return new_config

def autocorr(series, hist = 100):
    mu = np.mean(series)
    var = np.var(series)
    value = np.empty(hist)
    error = np.empty(hist)
    N_samples= series.size
    
    for k in range(hist):
        # ~ value[k] = np.mean((series[:-hist]-mu) *(series[k:k-hist]-mu))
        value[k] = np.mean((series[:N_samples-k]-mu) *(series[k:]-mu))
        
        # ~ error[k] = np.mean((series[:N_samples-k]-mu)**2 *(series[k:]-mu)**2 ) - value[k]**2
        error[k] = np.var( (series[:N_samples-k]-mu)*(series[k:]-mu) )
        error[k]=  np.sqrt( error[k]/(N_samples-k) )

    return value/var, error/var

def autocorr2(series, hist = 100):
    mu = np.mean(series)
    var = np.var(series)
    value = np.empty(hist)
    error = np.empty(hist)
    N_samples= series.size
    if var ==0:
       var= 1e-4
       
    
    for k in range(hist):
        # ~ value[k] = np.mean((series[:-hist]-mu) *(series[k:k-hist]-mu))
        value[k] = np.mean((series[:N_samples-k]-mu) *(series[k:]-mu))
        

    return value/var



def two_exp_model(t , c0, inv_tau0, c1, inv_tau1):
    return one_exp_model(t , c0, inv_tau0 ) + one_exp_model(t , c1, inv_tau1 )

def two_exp_model_simp(t , c0, inv_tau0, c1, inv_tau1):
    return one_exp_model(t , c0, inv_tau0 ) + one_exp_model(t , 1-c0, inv_tau1 )

def three_exp_model(t ,  c0, inv_tau0, c1, inv_tau1, c2, inv_tau2 ):
    return one_exp_model(t , c0, inv_tau0 ) + one_exp_model(t , c1, inv_tau1 ) + one_exp_model(t , c2, inv_tau2 ) 

def one_exp_model(t , c0, inv_tau0 ):
    return c0* np.exp(- inv_tau0* t, dtype='float64') 


def metropolis(beta, list_energy, list_log_prob):
    N_samples = list_energy.size
    assert list_log_prob.size==N_samples
    
    random_numbers = np.random.random(N_samples)
    accept_cont=[]

    for sample_numb in range(1,N_samples):
                    
        arg_exp = list_log_prob[sample_numb-1] - list_log_prob[sample_numb]+ beta*(list_energy[sample_numb-1]-list_energy[sample_numb])   
        if  random_numbers[sample_numb]> np.exp(arg_exp, dtype='float64'):  ## condition for rejecting configuration
            list_log_prob[sample_numb] = list_log_prob[sample_numb-1] 
            list_energy[sample_numb] = list_energy[sample_numb-1]
            accept_cont.append(0)
        else:
            accept_cont.append(1)            

    accept_cont = np.array(accept_cont)
    
    return list_energy, list_log_prob, accept_cont
      
    
                        
def fitting_tau(fitting_range, Gamma_cont, er_Gamma_cont):   
   
    ydata = Gamma_cont[fitting_range[0]:fitting_range[1] ]
    error_ydata= er_Gamma_cont[fitting_range[0]:fitting_range[1] ]
    xdata= np.arange(fitting_range[0],fitting_range[1])
    
    parameters, pcov = curve_fit(two_exp_model, xdata , ydata, sigma=error_ydata, bounds=([0.,1e-5,0.,1e-5], [1., 20., 1.,20.]) )   
    perr = np.sqrt(np.diag(pcov))

    c_tau_param = parameters.copy().reshape((2,2))
    c_tau_param[:,1] = 1/c_tau_param[:,1]

    perr = perr.reshape((2,2))
    perr= perr[:,1]*c_tau_param[:,1]**2
    perr = perr.reshape((2,1))
   
    c_tau_param = np.concatenate((c_tau_param, perr), axis=1)

    dtype = [('c', float ), ('tau', float), ('tau_err', float)]
    values = [(c_tau_param[0,0], c_tau_param[0,1],c_tau_param[0,2]), (c_tau_param[1,0], c_tau_param[1,1], c_tau_param[1,2]) ]
    c_tau_param = np.array(values, dtype=dtype) 
    c_tau_param = np.sort(c_tau_param, order='tau')   
    
    print( ' c0: ', c_tau_param[-1]['c'] ,' tau0: ',c_tau_param[-1]['tau'], '(',c_tau_param[-1]['tau_err'], ')',
           ' c1: ', c_tau_param[-2]['c'], ' tau1: ', c_tau_param[-2]['tau'], '(',c_tau_param[-2]['tau_err'], ')'  )

    
    return parameters, c_tau_param[-1]['tau'], c_tau_param[-1]['tau_err']


def plot_autocorrel_function(fitting_range,  parameters, Gamma_cont, er_Gamma_cont):

    xdata= np.arange(fitting_range[0],fitting_range[1])     
    param2 = parameters
    plt.plot(xdata, two_exp_model(xdata, *param2), 'r-')


    plt.errorbar(np.arange(Gamma_cont.size), Gamma_cont, yerr=er_Gamma_cont, fmt='.b')

    # ~ print(Gamma_cont)

    plt.show()        


def sample_van_sequence(net, n_samples):
    samples = torch.zeros([n_samples, 1, net.L, net.L])
    
    for step in range(n_samples):
        with torch.no_grad():
            # Use fast_sampling for VAN class or sample for others
            if hasattr(net, 'fast_sampling'):
                sample, _ = net.fast_sampling(1)
            else:
                sample, _ = net.sample(1)
        samples[step] = sample
    
    return samples


def analyze_van_autocorr(samples, ham, lattice, boundary, L):

    import ising
    
    energies = []
    mags = []
    
    for sample in samples:
        energy = ising.energy(sample.unsqueeze(0), ham, lattice, boundary).item() / L**2
        mag = sample.mean().item()
        energies.append(energy)
        mags.append(mag)
    
    energies = np.array(energies)
    mags = np.array(mags)
    
    hist_len = min(100, len(energies)//4)
    gamma_E, err_gamma_E = autocorr(energies, hist=hist_len)
    gamma_M, err_gamma_M = autocorr(mags, hist=hist_len)
    
    fitting_range = [2, min(50, len(energies)//8)]
    
    results = {
        'gamma_E': gamma_E, 'err_gamma_E': err_gamma_E,
        'gamma_M': gamma_M, 'err_gamma_M': err_gamma_M,
        'energies': energies, 'mags': mags
    }
    
    try:
        _, tau_E, tau_E_err = fitting_tau(fitting_range, gamma_E, err_gamma_E)
        results['tau_E'] = tau_E
        results['tau_E_err'] = tau_E_err
        print(f"Energy autocorr time: {tau_E:.2f} ± {tau_E_err:.2f}")
    except:
        results['tau_E'] = np.nan
        results['tau_E_err'] = np.nan
        print("Energy autocorr fit failed")
    
    try:
        _, tau_M, tau_M_err = fitting_tau(fitting_range, gamma_M, err_gamma_M)
        results['tau_M'] = tau_M
        results['tau_M_err'] = tau_M_err
        print(f"Magnetization autocorr time: {tau_M:.2f} ± {tau_M_err:.2f}")
    except:
        results['tau_M'] = np.nan
        results['tau_M_err'] = np.nan
        print("Magnetization autocorr fit failed")
    
    return results