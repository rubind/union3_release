# union3_release

Running UNITY1.5 requires:

numpy==1.22.4 pystan==2.19.1.1 cython==3.0.10

# Running UNITY1.5 with Union3:

```python read_and_sample.py inputs_Amanullah10_CNIa02_CSP_CalanTololo_CfA1_CfA2_CfA3_CfA4_DES3_Deep_DES3_Shallow_ESSENCE_Foundation_LOSS_MCT_NB99_Pan-STARRS_Riess07_SDSS_SNLS_SuzukiRubin_Tonry03_LSQ+LCO_LSQ_knop03_Krisciunas.pickle [cosmo_model, e.g., 1]```

available cosmology models are 1 (flat LCDM), 2 (Spline-interpolated distance modulus), 3 (flat Om-w), 4 (q0-j0), 5 (flat Om-w0-wa)

It will take hours to run and use many GB of RAM.

# Files:

* inputs_Amanullah10_CNIa02_CSP_CalanTololo_CfA1_CfA2_CfA3_CfA4_DES3_Deep_DES3_Shallow_ESSENCE_Foundation_LOSS_MCT_NB99_Pan-STARRS_Riess07_SDSS_SNLS_SuzukiRubin_Tonry03_LSQ+LCO_LSQ_knop03_Krisciunas.pickle pickle file containing Union3
* read_and_sample.py python script that reads in data and calls UNITY
* stan_code_simple.txt nominal UNITY1.5 model
* stan_code_fixed.txt model with improved outlier limits (see Appendix B)
* mu_mat_union3_cosmo=2_mu.fits compressed distances for Union3+UNITY1.5; the first row is redshift, the first column is distance modulus, the rest of the matrix is the inverse covariance matrix
* lcfit_Union3.tar.gz is a tarball of input files, lc fitting result files, and SNe passing cuts (*_v1.txt). Note that the MW extinction is double scaled by 0.86, but UNITY self-calibrates this out so it ends up not mattering.
* simple_Gaussian_check.py reads all_samples_union3_cosmo=2.npz and shows some summary statistics on the spline nodes (all_samples_union3_cosmo=2.npz are the spline-node MCMC samples).
