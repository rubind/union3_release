from numpy import *
import numpy as np
from scipy.stats import scoreatpercentile
from scipy.interpolate import interp1d, RectBivariateSpline
import os
import subprocess
from astropy.io import fits

################################################# File-Reading Functions ###################################################
def clean_lines(lines, stringlist = [""]):
    # Start by getting rid of spaces.

    lines = [item.strip() for item in lines]
    
    # Check for strings to exclude.
    lines = [item for item in lines if stringlist.count(item) == 0]

    # Get rid of comments
    lines = [item for item in lines if item[0] != "#"]
    return lines

    

def read_param(flnm, param, default = None, ind = 1):
    # This one is commonly used for photometry parameter files.
    
    fp = open(flnm)
    lines = fp.read()
    fp.close()

    lines = lines.split('\n')
    lines = clean_lines(lines)

    for line in lines:
        parsed = line.split(None)
        if parsed[0] == param:
            print("Reading " + param + " from " + flnm)

            try:
                # Yeah, I know eval is bad. But it works with all types!
                return eval(parsed[ind])
            except:
                return parsed[ind]
    print() 
    print("Couldn't find ", param, flnm)
    print("Returning default ", default)
    print()

    return default


def get_params(paramfl):
    print("Reading params from ", paramfl)

    f = open(paramfl)
    lines = f.read().split('\n')
    f.close()

    keys = ["filenamelist", "weird_sn_list", "mag_cut", "calibration_uncertainties",
            "iter", "chains", "n_jobs",
            "max_firstphase", "min_lastphase", "max_color_uncertainty", "max_color", "min_color", "max_MWEBV",
            "min_redshift", "max_redshift", "redshift_coeff_type",
            "do_blind", "do_twoalphabeta", "separate_mass_x1c", "outl_frac", "remap_x1",
            "stan_code", "pec_vel_disp", "lensing_disp", "MWEBV_zeropoint_EBV", "electron_coeff", "IG_extinction_coeff",
            "sample_file", "max_params_to_save",
            "do_host_mass", "fix_Om", "MB_by_sample", "include_pec_cov", "threeD_unexplained"
            ]
    
    params = {}

    for line in lines:
        parsed = line.split(None)
        if len(parsed) > 1:
            for key in keys:
                if parsed[0] == key:
                    if len(parsed) == 2:
                        try:
                            params[key] = eval(parsed[1])
                        except:
                            params[key] = parsed[1]
                    else:
                        params[key] = parsed[1:]

    for key in keys:
        if key not in params:
            print("Didn't read ", key)
            assert 0
            

    if params["sample_file"].count("None"):
        params["sample_file"] = None
    else:
        params["sample_file"] = os.getcwd() + "/" + params["sample_file"]
            
    for key in ["weird_sn_list", "mag_cut"]:
        if params[key].count("$"):
            print(params[key])
            params[key] = subprocess.getoutput("ls " + params[key])
            print("->", params[key])
    for i in range(len(params["filenamelist"])):
        if params["filenamelist"][i].count("$"):
            print(params["filenamelist"][i])
            params["filenamelist"][i] = subprocess.getoutput("ls " + params["filenamelist"][i])
            print("->", params["filenamelist"][i])

    if not isinstance(params["min_redshift"], list):
        params["min_redshift"] = [params["min_redshift"]] * len(params["filenamelist"])

    if not isinstance(params["max_redshift"], list):
        params["max_redshift"] = [params["max_redshift"]] * len(params["filenamelist"])

    print("Read params ", params)
    assert isinstance(params["filenamelist"], list), "filenamelist should be a list!"

    f = fits.open(os.environ["UNITY"] + "/paramfiles/Azwave_grid.fits")
    IG_dat = f[0].data
    f.close()

    params["IG_fn"] = RectBivariateSpline(IG_dat[1:, 0],
                                          IG_dat[0, 1:],
                                          IG_dat[1:, 1:], kx = 1, ky = 1)
    
    return params



def get_dparam_dzps(res_der_fl, redshift, calibration_paths):
    wavebins = [(3000.0, 4000.0),
                (4000.0, 5000.0),
                (6000.0, 8000.0),
                (8000.0, 100000.0),
                (10000.0, 100000.0)] # 4000 and 8000 for the breaks, roughly, 10000 for STIS -> WFC3/NICMOS

    Landolt_Smith_bins = dict(L = {"LANDOLT_U": (3000, 4000),
                                   "LANDOLT_B": (4000, 5000),
                                   "LANDOLT_V": (5000, 6000),
                                   "LANDOLT_R": (6000, 7500),
                                   "LANDOLT_I": (7500, 9000)},
                              S = {"SMITH_u": (3000, 4000),
                                   "SMITH_g": (4000, 5500),
                                   "SMITH_r": (5500, 7000),
                                   "SMITH_i": (7000, 8000),
                                   "SMITH_z": (8000, 10000)},
                              P = {"PS1_g": (4000, 5500),
                                   "PS1_r": (5500, 7000),
                                   "PS1_i": (7000, 8000),
                                   "PS1_z": (8000, 10000)}
                              )
    
    
    f = open(res_der_fl)
    lines = f.read().split('\n')
    f.close()

    dparam_dzps = {}
    for line in lines:
        parsed = line.split(None)
        if parsed.count("All") and (parsed.count("Zeropoint") or parsed.count("Lambda")):
            restlamb = float(parsed[2])
            obslamb = (1. + redshift)*restlamb

            parsed = line.split(None)
            assert parsed[3] == "All", "Weird format for " + res_der_fl

            the_key = (parsed[0], parsed[1][parsed[1].find("|")+1:])
            dparam_dzps[the_key] = array([float(parsed[5]), float(parsed[6]), float(parsed[7])])


            for Landolt_or_Smith in "LSP":
                if calibration_paths[the_key] == Landolt_or_Smith:
                    for LSkey in Landolt_Smith_bins[Landolt_or_Smith]:
                        if (Landolt_Smith_bins[Landolt_or_Smith][LSkey][0] < obslamb) and (Landolt_Smith_bins[Landolt_or_Smith][LSkey][1] > obslamb):
                            if parsed.count("Zeropoint"):
                                key_to_write = LSkey + "_CAL"
                            else:
                                key_to_write = LSkey + "_LAM"

                            if key_to_write in dparam_dzps:
                                dparam_dzps[key_to_write] += array([float(parsed[5]), float(parsed[6]), float(parsed[7])])
                            else:
                                dparam_dzps[key_to_write] = array([float(parsed[5]), float(parsed[6]), float(parsed[7])])

                                                    
            
        if parsed.count("Zeropoint") and parsed.count("All"):
            restlamb = float(parsed[2])
            obslamb = (1. + redshift)*restlamb

            for wavebin in wavebins:
                if (obslamb >= wavebin[0]) and (obslamb < wavebin[1]):
                    thekey = ("Fundamental", wavebin)

                    if thekey in dparam_dzps:
                        dparam_dzps[thekey] += array([float(parsed[5]), float(parsed[6]), float(parsed[7])])
                    else:
                        dparam_dzps[thekey] = array([float(parsed[5]), float(parsed[6]), float(parsed[7])])

            if restlamb < 3400:
                thekey = "SALT_UV_CAL"
                
                if thekey in dparam_dzps:
                    dparam_dzps[thekey] += array([float(parsed[5]), float(parsed[6]), float(parsed[7])])
                else:
                    dparam_dzps[thekey] = array([float(parsed[5]), float(parsed[6]), float(parsed[7])])

            if restlamb < 4000:
                thekey = "SALT_U_CAL"
                
                if thekey in dparam_dzps:
                    dparam_dzps[thekey] += array([float(parsed[5]), float(parsed[6]), float(parsed[7])])
                else:
                    dparam_dzps[thekey] = array([float(parsed[5]), float(parsed[6]), float(parsed[7])])
                
            if restlamb > 7000:
                thekey = "SALT_I_CAL"
                
                if thekey in dparam_dzps:
                    dparam_dzps[thekey] += array([float(parsed[5]), float(parsed[6]), float(parsed[7])])
                else:
                    dparam_dzps[thekey] = array([float(parsed[5]), float(parsed[6]), float(parsed[7])])
                


    for key in dparam_dzps:
        print("dparam_dzps", key, dparam_dzps[key])
        

    
    return dparam_dzps

def get_MWEBV_uncs(lightfl, res_der_fl, params):

    sig_stat = 0.16    # 16% statistical uncertainty
    sig_norm = 0.10    # 10% multiplicative normalization uncertainty
    sig_add  = params["MWEBV_zeropoint_EBV"]   # E.g., 5 mmag E(B-V) additive uncertainty

    d_dMWEBV = array([read_param(res_der_fl, "MWEBV", ind = 5),
                      read_param(res_der_fl, "MWEBV", ind = 6),
                      read_param(res_der_fl, "MWEBV", ind = 7)])

    # Add support for pre-correction for extinction in error propagation
    MWEBV = read_param(lightfl, "MW_true_EBV")
    if MWEBV == None:
        MWEBV = read_param(lightfl, "MWEBV")

    dparam_dzps = {"MWEBV_multnorm": MWEBV*sig_norm*d_dMWEBV, "MWEBV_addnorm": sig_add*d_dMWEBV}
    extra_cmat = outer(MWEBV*sig_stat*d_dMWEBV, MWEBV*sig_stat*d_dMWEBV) # Has the same mB, x1, c order as LC covariance matrices

    return dparam_dzps, extra_cmat

def get_electron_scattering(redshift, params):
    # Nominally 0.0042  0.00042
    assert abs(params["electron_coeff"][0]) < 0.01
    assert abs(params["electron_coeff"][1]) < 0.01
    assert params["electron_coeff"][0] > -0.001

    tau_electron = params["electron_coeff"][0] * (sqrt(0.3*(1 + redshift)**3. + 0.7) - 1.)
    dtau_electron = params["electron_coeff"][1] * (sqrt(0.3*(1 + redshift)**3. + 0.7) - 1.)

    add_mag_electron = -2.5/log(10.) * tau_electron
    dmag_electron = 2.5/log(10.) * dtau_electron

    dparam_dzps = {"electron_scattering": array([dmag_electron, 0, 0])}

    return dparam_dzps, add_mag_electron

def eval_IG_extinction(z, efflambrest_A, params):    
    """Menard model
test: eval_IG_extinction(1.0, 4400.) 0.020019767101606717
"""    
    efflamb = efflambrest_A/10000.
    return params["IG_fn"](z, efflamb)[0,0]
    


def get_IG_extinction_sys(redshift, res_der_fl, params):
    assert eval_IG_extinction(1.0, 4400, params) > eval_IG_extinction(0.5, 4400, params)
    assert eval_IG_extinction(1.0, 4400, params) > eval_IG_extinction(1.0, 5500, params)

    
    f = open(res_der_fl)
    lines = f.read().split('\n')
    f.close()

    dparam_dzps = {"IG_extinction": np.zeros(3, dtype=np.float64)}
    
    for line in lines:
        parsed = line.split(None)
        if parsed.count("Zeropoint"):
            IGextinct = eval_IG_extinction(redshift, float(parsed[2]), params)*params["IG_extinction_coeff"]

            this_dmu_zp = np.array([float(parsed[5]), float(parsed[6]), float(parsed[7])])*IGextinct
            dparam_dzps["IG_extinction"] += this_dmu_zp
    return dparam_dzps


def get_lensing_bias(redshift, lensing_ifn):
    dparam_dzps = {"lensing_bias": array([lensing_ifn(redshift), 0, 0])}

    return dparam_dzps

            

def get_calib_uncertainties(calib_names, zeropointfl):
    assert 0, "Deprecated!"

    calib_uncertainties = []

    f = open(zeropointfl)
    lines = f.read().split('\n')
    f.close()

    print(lines)
    lkfjds

    
    for calib_name in calib_names:
        found = 0

        for line in lines:
            parsed = line.split(None)

            if parsed.count(calib_name[1]):
                if calib_name[0] == "Zeropoint":
                    calib_uncertainty = float(parsed[1])
                elif calib_name[0] == "Lambda":
                    calib_uncertainty = float(parsed[2])
                print("Applying ", calib_uncertainty, "for", calib_name)
                calib_uncertainties.append(calib_uncertainty)
                found += 1
        assert found == 1, "Found zero times or more than once! " + str(calib_name) + " " + str(found)

    return calib_uncertainties

def merge_calib(the_data, dparam_dzps, current_sn_ind, uncertainties, check_1 = False):
    for key in dparam_dzps:
        if not the_data["calib_names"].count(key):
            the_data["calib_names"].append(key)
            
        calib_ind = the_data["calib_names"].index(key)
        the_data["d_mBx1c_dcalib_list"][current_sn_ind, :, calib_ind] = dparam_dzps[key]*uncertainties[key]

        if check_1:
            assert uncertainties[key] == 1
            
    return the_data
    

def samples_txt_to_pickle(flname, samples_to_burn, skip = 6):
    """NOTE: these samples come out in a different order than the extracted fit!"""

    f = open(flname)
    lines = f.read().split('\n')
    f.close()

    headings_size = {}
    headings = []

    for i in range(6):
        parsed = lines[i].split(",")[skip:]
        if len(parsed) > 1 and len(list(headings_size.keys())) == 0:
            headings = parsed
            for j in range(len(parsed)):
                pparsed = parsed[j].split(".")
                new_key = pparsed[0] not in headings_size

                if parsed[j].count(".") == 0:
                    headings_size[pparsed[0]] = 0
                elif parsed[j].count(".") == 1:
                    if new_key:
                        headings_size[pparsed[0]] = (int(pparsed[1]),)
                    else:
                        headings_size[pparsed[0]] = (max(headings_size[pparsed[0]][0], int(pparsed[1])),
                                                     )
                elif parsed[j].count(".") == 2:
                    if new_key:
                        headings_size[pparsed[0]] = (int(pparsed[1]), int(pparsed[2]))
                    else:
                        headings_size[pparsed[0]] = (max(headings_size[pparsed[0]][0], int(pparsed[1])),
                                                     max(headings_size[pparsed[0]][1], int(pparsed[2]))
                                                     )
                elif parsed[j].count(".") == 3:
                    if new_key:
                        headings_size[pparsed[0]] = (int(pparsed[1]), int(pparsed[2]), int(pparsed[3]))
                    else:
                        headings_size[pparsed[0]] = (max(headings_size[pparsed[0]][0], int(pparsed[1])),
                                                     max(headings_size[pparsed[0]][1], int(pparsed[2])),
                                                     max(headings_size[pparsed[0]][2], int(pparsed[3]))
                                                     )
                else:
                        assert 0, "!!!!!!" + parsed[j]

    samples_count = -samples_to_burn
    
    for line in lines:
        parsed = line.split(",")[skip:]
        if len(parsed) == len(headings):
            try:
                float(parsed[0])
                samples_count += 1
            except:
                pass

    print("samples_count ", samples_count)

    print(headings_size)
    samples = {}

    for key in headings_size:
        if headings_size[key] == 0:
            samples[key] = zeros(samples_count, dtype=float64) - 1.e20
        else:
            samples[key] = zeros((samples_count,) + headings_size[key], dtype=float64)  - 1.e20
                    
    linecount = -samples_to_burn - 1
    for line in lines:
        parsed = line.split(",")[skip:]

        if len(parsed) == len(headings):
            try:
                float(parsed[0])
                skip_line = 0
            except:
                skip_line = 1

            if not skip_line:
                linecount += 1
                if linecount >= 0:
                    for j in range(len(parsed)):
                        pparsed = headings[j].split(".")
                        
                        if len(pparsed) == 1:
                            samples[pparsed[0]][linecount] = float(parsed[j])
                        else:
                            key = tuple([linecount] + [int(item) - 1 for item in pparsed[1:]])
                            samples[pparsed[0]][key] = float(parsed[j])
    return samples


def get_kcorrect_ifns(magcut_k_correction_fl):
    from FileRead import readcol

    [z, c2, c3] = readcol(magcut_k_correction_fl, 'fff')
    return interp1d(z, c2, kind = 'linear'), interp1d(z, c3, kind = 'linear')

################################################# x1 Remapping ###################################################

def remap_x1(x1, params):
    """params["remap_x1"] is negative. This mapping moves large x1 values towards zero and small x1 values further from zero. The slope is the slope of the mapping (smaller for larger x1). The off-diagonal x1 covariances scale by the slope (e.g., the precision improves for large x1s and decreases for small x1s); the x1-x1 covariance scales by the slope**2."""
    
    new_x1 = x1 + float(params["remap_x1"][0]) * x1**2. + float(params["remap_x1"][1]) * x1**3.
    x1_slope = 1. + 2.*float(params["remap_x1"][0]) * x1 + 3*float(params["remap_x1"][1]) * x1**2.

    return new_x1, x1_slope
    

################################################# Chain Functions ###################################################

def gelman_rubin_R(samples):
    """samples should be an array (nsamples = n, nchains = m). This is the original formula without the sqrt!"""
    n = len(samples)
    print("nsamples ", n)
    m = len(samples[0])
    print("nchains ", m)

    psi_dotj = mean(samples, axis = 0)
    psi_dotdot = mean(psi_dotj)
    s_j2 = 1./(n - 1.) * sum((samples - psi_dotj)**2.)

    B = n/(m - 1.) * sum((psi_dotj - psi_dotdot)**2.)
    W = (1./m)*sum(s_j2)

    var_t = (n - 1.)*W/n + B/n
    return var_t/W



def filter_fit_params(fit_params, param_name, chains, iter_per_chain):

    stdevs = array([], dtype=float64)
    for i in range(chains):
        stdevs = append(stdevs, std(fit_params[param_name][i*iter_per_chain:(i+1)*iter_per_chain])
                        )
    print(param_name, "stdevs ", stdevs)
    good_chains = stdevs > max(stdevs)/4.
    print("good_chains", good_chains)
    
    good_inds = []
    for i in range(chains):
        if good_chains[i]:
            good_inds.extend(list(range(i*iter_per_chain, (i+1)*iter_per_chain)))

    for key in fit_params:
        print(key, fit_params[key].shape, end=' ')
        try:
            fit_params[key] = fit_params[key][good_inds]
        except:
            print("Error!")
            sys.exit(1)
        print(fit_params[key].shape)
    return fit_params


def quick_print(vals, thename):
    txt = "  ".join(["SummaryPrint ", thename, "Mean 15.9 50 84.1", "%.4f" % mean(vals), "%.4f" % scoreatpercentile(vals, 15.8655), "%.4f" % scoreatpercentile(vals, 50), "%.4f" % scoreatpercentile(vals, 84.1345)])
    return txt


def summarize_parameters(fit_params, thekeys = None, on_screen = 1):
    if thekeys == None:
        thekeys = list(fit_params.keys())

    txt = ""

    for parameter in thekeys:
        the_shape = fit_params[parameter].shape

        if len(the_shape) == 1:
            txt += quick_print(fit_params[parameter], parameter) + '\n'
        elif len(the_shape) == 2:
            for i in range(the_shape[1]):
                txt += quick_print(fit_params[parameter][:,i], parameter + "::" + str(i)) + '\n'
        elif len(the_shape) == 3:
            for i in range(the_shape[1]):
                for j in range(the_shape[2]):
                    txt += quick_print(fit_params[parameter][:,i,j], parameter + "::" + str(i) + "::" + str(j)) + '\n'
        else:
            pass
    
    if on_screen:
        print(txt)
    return txt

################################################# Plotting Functions ###################################################

def short_year(txt):
    for i in range(1800, 2200):
        txt = txt.replace(str(i), str(i)[2:])
    return txt




def bin_samples_in_redshift(stan_data, the_data, redshift_bins = np.array([0.01, 0.15, 0.95, 1.65])):
    f = open(os.environ["UNITY"] + "/paramfiles/sample_names_colors.txt", 'r')
    lines = f.read()
    f.close()

    for i in range(20):
        lines = lines.replace("\t\t", "\t").replace(" \t", "\t").replace("\t ", "\t").replace("  ", " ")
    
    lines = lines.split('\n')

    for i in range(len(lines))[::-1]:
        if lines[i].count('\t') == 2:
            lines[i] = lines[i].split('\t')
        else:
            if len(lines[i].split(None)) > 1:
                print("Skipping ", lines[i])
            del lines[i]

    assert np.min(stan_data["sample_list"]) == 1
    assert np.max(stan_data["sample_list"]) == stan_data["n_samples"]

    samp_bins = dict(cats = [], colors = [], labels = [], short_labels = [], inds = [])

    for samp_cat_target in range(len(redshift_bins)):
        for i in range(stan_data["n_samples"]):
            inds = where(stan_data["sample_list"] == i + 1)
            zmax = max(stan_data["redshifts"][inds])
            samp_cat = argmin(abs(zmax - redshift_bins))
            
            if samp_cat == samp_cat_target:
                ind = [item[0] for item in lines].index(the_data["sample_names"][i].split("/")[-1])

                samp_bins["cats"].append(samp_cat)
                samp_bins["labels"].append(
                    lines[ind][1].replace("_", "\n"))
                short_label = lines[ind][1].replace("_", " ")
                short_label = short_label.split(None)
                print("short_label", short_label)
                
                for j in range(1, len(short_label))[::-1]:
                    if short_label[j][-1] == "+":
                        if short_label[j][0] == "(":
                            offset = 1
                        else:
                            offset = 0
                            
                        short_label[j] = short_label[j][:offset+1] + short_year(short_label[j+1])
                        del short_label[j+1]
                        
                short_label = " ".join([short_year(item) for item in short_label])
                
                samp_bins["short_labels"].append(short_label)
                samp_bins["colors"].append(eval(lines[ind][2]))
                samp_bins["inds"].append(i)
    return samp_bins

