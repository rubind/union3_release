import pickle
import gzip
import numpy as np
from scipy.stats import scoreatpercentile

loaded = np.load("all_samples_union3_cosmo=2.npz")
mu_znodes = loaded["arr"]

    
mu_znodes = (mu_znodes.T - np.median(mu_znodes, axis = 1)).T # Subtract off median for each sample, as this is degenerate with scriptM

    
for i in range(len(mu_znodes[0])):
    mu_samples = mu_znodes[:,i] # Samples for this bin
    mean_mu = np.mean(mu_samples)
    std_mu = np.std(mu_samples, ddof=1)
    pull_samples = (mu_samples - mean_mu)/std_mu

    percentile_txt = []
    for percentile in [2.27501, 15.8655, 50., 84.1345, 97.725]:
        percentile_txt.append("%.1fth: %.2f" % (percentile, scoreatpercentile(pull_samples, percentile)))
    
    print("spline node %i: pull percentiles: %s" % (i, ", ".join(percentile_txt)))
    

    
