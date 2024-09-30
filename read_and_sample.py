from matplotlib import use
use("PDF")
import pickle
from numpy import *
import numpy as np
import multiprocessing
multiprocessing.set_start_method("fork")

import pystan
import sys
import os
sys.path.append(os.environ["UNITY"] + "/other_cosmology/")

from cosmo_functions import get_mu

import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
import helper_functions
import Spectra
from scipy.interpolate import interp1d
import gzip
from FileRead import readcol
from astropy.io import fits
from DavidsNM import save_img#, miniNM_new
from scipy.special import erf


################################################# Get the SALT data ###################################################


def read_data(params):

    the_data = {"mB_list": array([], dtype=float64),
                "x1_list": array([], dtype=float64),
                "c_list": array([], dtype=float64),
                "mBx1c_cov_list": zeros([0,3,3], dtype=float64),
                "z_CMB_list": array([], dtype=float64),
                "z_helio_list": array([], dtype=float64),
                "sample_list": array([], dtype=int32), # SN sample, from 0 to N_samples - 1
                "sample_names": [], # For storing sample names
                "mag_cut_list": array([], dtype=float64),
                "mag_cut_disp_list": array([], dtype=float64), # Dispersion on magnitude cut
                "mass": [], # Host mass
                "mass_err": [], # Host-mass uncertainty
                "snpaths": [], # Paths to LC fits. Stored for future reference.
                "RA": [],
                "Dec": [],
                
                "mobs_cut0": [],
                "mobs_cut1": [],
                "est_mobs_cuts": [],
                "est_mobs_sigmas": [],
                
                "efflambs": {}, # Filter wavelengths
                "calib_names": [], # Name of each systematic uncertainty
                
                "d_mBx1c_dcalib_list": zeros([3000,3,1000], dtype=float64), # This is an inefficient way to do this, but this is initialized to fixed size, then trimmed later.

                "photoz_inds": [],
                "d_mBx1c_dz_list": [],
                "photo_z0": [],
                "photo_dz": [],
                "n_photoz": 0, # Number of SNe with photo-z's
                "photo_spikez": []
            }


    current_sn_ind = 0
    
    filenamelist = params["filenamelist"]

    f_read = open("sn_input.txt", 'w')
    f_read.write("#SN\tRA\tDEC\tZHEL\tZCMB\tPASS\n") # List of SNe that pass all cuts

    [magcut_input_fls, magcut_k_correction_fls, magcut_est_cuts, magcut_est_sigmas] = readcol(params["mag_cut"], 'aaff')
    magcut_k_correction_fls = [item.replace("$UNITY", os.environ["UNITY"]) for item in magcut_k_correction_fls]
    
    [bulk_RA, bulk_Dec, bulk_z] = readcol(os.environ["UNITY"] + "/paramfiles/table.input2", 'fff')
    fbulk = fits.open(os.environ["UNITY"] + "/paramfiles/dominant_evecs.fits")
    bulk_eig = fbulk[0].data
    fbulk.close()


    [lensing_z, lensing_mag] = readcol(os.environ["UNITY"] + "/paramfiles/lensing_bias.txt", 'ff')
    lensing_mag = 0.5*(0.055*lensing_z)**2.
    print("LENSING HACK!!!!!")
    lensing_ifn = interp1d(lensing_z, lensing_mag, kind = 'linear')

    f_cal = open(params["calibration_uncertainties"].replace("$UNITY", os.environ["UNITY"]), 'r') #open(os.environ["UNITY"] + "/paramfiles/calibration_uncertainties.txt", 'r')
    lines = f_cal.read().split('\n')
    f_cal.close()
    calibration_uncertainties = {}
    calibration_paths = {}
    
    for line in lines:
        parsed = line.split(":")
        if len(parsed) > 1:
            parsed = [item.strip() for item in parsed]

            calibration_paths[eval(parsed[0])] = "None"

            for possible_path in "LSP":
                if parsed[1].count(possible_path):
                    calibration_paths[eval(parsed[0])] = possible_path
                    parsed[1] = parsed[1].replace(possible_path, "")

                
            calibration_uncertainties[eval(parsed[0])] = float(parsed[1])
    

    print("calibration_paths", calibration_paths)
    
    calibration_uncertainties["MWEBV_multnorm"] = 1.
    calibration_uncertainties["MWEBV_addnorm"] = 1.
    calibration_uncertainties["electron_scattering"] = 1.
    calibration_uncertainties["IG_extinction"] = 1.
    calibration_uncertainties["lensing_bias"] = 1.

    for key in calibration_uncertainties:
        print("calibration_uncertainties", key, calibration_uncertainties[key])
                                      
    assert len(bulk_eig[0]) == len(bulk_RA)

    for current_sample, directory in enumerate(filenamelist):
        the_data["sample_names"].append(directory)

        f = open(directory)
        snpaths = f.read().split('\n')
        snpaths = [item.strip() for item in snpaths]
        snpaths = [item for item in snpaths if item != ""]
        snpaths = [item.replace("$UNION", os.environ["UNION"]) for item in snpaths]

        f.close()

        print("current_sample, directory", current_sample, directory)
        magcut_ind = magcut_input_fls.index(directory.split("/")[-1])
        print("magcut_ind ", magcut_ind)

        the_data["est_mobs_cuts"].append(magcut_est_cuts[magcut_ind])
        the_data["est_mobs_sigmas"].append(magcut_est_sigmas[magcut_ind])
        
        kc_ifn0, kc_ifn1 = helper_functions.get_kcorrect_ifns(magcut_k_correction_fls[magcut_ind])

        for snpath in snpaths:
            this_redshift_cmb = helper_functions.read_param(snpath + "/lightfile", "z_cmb")
            if this_redshift_cmb is None:
                this_redshift_cmb = helper_functions.read_param(snpath + "/lightfile", "z_CMB")

            this_redshift_helio = helper_functions.read_param(snpath + "/lightfile", "z_heliocentric")
            this_redshift_helio_true = helper_functions.read_param(snpath + "/lightfile", "z_true_heliocentric")

            if this_redshift_helio_true is not None:
                this_redshift_helio = this_redshift_helio_true
            if this_redshift_cmb is None and this_redshift_helio > 0.1:
                this_redshift_cmb = this_redshift_helio
                print("Couldn't find redshift for ", snpath)
            if this_redshift_helio is None and this_redshift_cmb > 0.1:
                this_redshift_helio = this_redshift_cmb
                print("Couldn't find redshift for ", snpath)
                
            this_RA = helper_functions.read_param(snpath + "/lightfile", "RA")
            this_Dec = helper_functions.read_param(snpath + "/lightfile", "DEC")
            if this_Dec == None:
                this_Dec = helper_functions.read_param(snpath + "/lightfile", "Dec")

            this_firstphase = helper_functions.read_param(snpath + "/result_salt2.dat", "FirstPhase")
            this_lastphase = helper_functions.read_param(snpath + "/result_salt2.dat", "LastPhase")
            this_color = helper_functions.read_param(snpath + "/result_salt2.dat", "Color")
            this_colorerr = helper_functions.read_param(snpath + "/result_salt2.dat", "Color", ind = 2)
            this_x1 = helper_functions.read_param(snpath + "/result_salt2.dat", "X1", ind = 1)
            this_x1_err = helper_functions.read_param(snpath + "/result_salt2.dat", "X1", ind = 2)

            this_check = Spectra.check_derivs(snpath)


            if this_x1_err == None:
                this_x1 = 100.
                this_x1_err = 100.

            weird_sn = helper_functions.read_param(params["weird_sn_list"], snpath.split("/")[-1])
            print("weird_sn ", snpath, weird_sn)
            
            
            this_MWEBV = helper_functions.read_param(snpath + "/lightfile", "MW_true_EBV")
            if this_MWEBV == None:
                this_MWEBV = helper_functions.read_param(snpath + "/lightfile", "MWEBV")
            else:
                print("Using MW_true_EBV")
            print("this_MWEBV", this_MWEBV)

            
            okay_to_add = [this_redshift_cmb >= params["min_redshift"][current_sample],
                           this_redshift_cmb <= params["max_redshift"][current_sample],
                           this_firstphase <= params["max_firstphase"],
                           this_lastphase >= params["min_lastphase"],
                           #this_firstphase + 10 < this_lastphase,
                           this_colorerr < params["max_color_uncertainty"],
                           this_color < params["max_color"],
                           this_color > params["min_color"],
                           this_MWEBV <= params["max_MWEBV"],
                           weird_sn == None, this_check, abs(this_x1) + this_x1_err < 5]
            okay_names = ["min_z", "max_z", "first_p", "last_p", "colorerr", "colorcut", "weirdsn", "converge", "x1"]

            f_read.write('\t'.join([
                "/".join(snpath.split("/")[-2:]),
                str(this_RA),
                str(this_Dec),
                str(this_redshift_helio),
                str(this_redshift_cmb),
                str(all(okay_to_add))]
                               ) + '\n')


            if all(okay_to_add):

                the_data["snpaths"].append(snpath)
                if Spectra.check_derivs(snpath, flname = "model_deriv.dat") == 0:
                    assert Spectra.check_derivs(snpath, flname = "result_deriv.dat") == 1
                    deriv_file_to_use = snpath + "/result_deriv.dat"
                    print("Using observed derivatives instead!", snpath)
                else:
                    deriv_file_to_use = snpath + "/model_deriv.dat"


                if helper_functions.read_param(snpath + "/lightfile", "Photoz") != None:
                    print("Photoz found!", snpath)
                    the_data["n_photoz"] += 1
                    
                    the_data["photoz_inds"].append(the_data["n_photoz"]) # That's right, after incrementing the counter
                    the_data["d_mBx1c_dz_list"].append([helper_functions.read_param(deriv_file_to_use, "Redshift", ind = 5),
                                                        helper_functions.read_param(deriv_file_to_use, "Redshift", ind = 6),
                                                        helper_functions.read_param(deriv_file_to_use, "Redshift", ind = 7)])

                    the_data["photo_z0"].append(helper_functions.read_param(snpath + "/lightfile", "Photoz", ind = 1))
                    the_data["photo_dz"].append(helper_functions.read_param(snpath + "/lightfile", "Photoz", ind = 2))
                    the_data["photo_spikez"].append(this_redshift_helio)
                    
                else:
                    the_data["photoz_inds"].append(0)


                the_data["z_CMB_list"] = append(the_data["z_CMB_list"], this_redshift_cmb
                                                )
                the_data["z_helio_list"] = append(the_data["z_helio_list"], this_redshift_helio
                                                )

                the_data["mobs_cut0"].append(kc_ifn0(this_redshift_helio))
                the_data["mobs_cut1"].append(kc_ifn1(this_redshift_helio))
                the_data["RA"].append(this_RA)
                the_data["Dec"].append(this_Dec)
                

                the_data["mB_list"] = append(the_data["mB_list"],
                                             helper_functions.read_param(snpath + "/result_salt2.dat", "RestFrameMag_0_B"))
                the_data["c_list"] = append(the_data["c_list"],
                                            helper_functions.read_param(snpath + "/result_salt2.dat", "Color"))
                the_data["x1_list"] = append(the_data["x1_list"],
                                             helper_functions.read_param(snpath + "/result_salt2.dat", "X1"))


                the_data["sample_list"] = append(the_data["sample_list"], current_sample)

                this_mass = helper_functions.read_param(snpath + "/lightfile", "Mass", ind = 1)
                this_mass_err = sqrt(abs(
                    helper_functions.read_param(snpath + "/lightfile", "Mass", ind = 2)*helper_functions.read_param(snpath + "/lightfile", "Mass", ind = 3)
                    ))

                if this_mass == None or this_mass < 1 or this_mass_err == 0. or isinf(this_mass_err):
                    if the_data["z_CMB_list"][-1] > 0.1:
                        the_data["mass"].append(10.)
                        the_data["mass_err"].append(1.)
                    else:
                        the_data["mass"].append(11.)
                        the_data["mass_err"].append(1.)
                else:
                    the_data["mass"].append(this_mass)
                    the_data["mass_err"].append(this_mass_err)


                # First term from SALT, second term from 300 km/s, third term lensing (may be overestimated)
                mBmB = helper_functions.read_param(snpath + "/result_salt2.dat", "RestFrameMag_0_B", ind = 2)**2. + (params["lensing_disp"]*the_data["z_CMB_list"][-1])**2.
                mBx1 = helper_functions.read_param(snpath + "/result_salt2.dat", "CovRestFrameMag_0_BX1")
                mBc = helper_functions.read_param(snpath + "/result_salt2.dat", "CovColorRestFrameMag_0_B")
                x1x1 = helper_functions.read_param(snpath + "/result_salt2.dat", "CovX1X1")
                x1c = helper_functions.read_param(snpath + "/result_salt2.dat", "CovColorX1")
                cc = helper_functions.read_param(snpath + "/result_salt2.dat", "CovColorColor")

                h_resid = (the_data["mB_list"][-1] - - 19.1 + 0.13*the_data["x1_list"][-1] - 3.*the_data["c_list"][-1]) - (5*log10(the_data["z_CMB_list"][-1]*(1. + the_data["z_helio_list"][-1])) + 42.9)
                if abs(h_resid) > 2 or (the_data["c_list"][-1] > 1) or (the_data["c_list"][-1] < -0.3):
                    print("Weird supernova!", snpath)

                dparam_dzps, extra_cmat = helper_functions.get_MWEBV_uncs(snpath + "/lightfile", res_der_fl = deriv_file_to_use, params = params)
                the_data = helper_functions.merge_calib(the_data = the_data, dparam_dzps = dparam_dzps, current_sn_ind = current_sn_ind, uncertainties = calibration_uncertainties, check_1 = True)

                if params["IG_extinction_coeff"] != 0:
                    dparam_dzps = helper_functions.get_IG_extinction_sys(redshift = the_data["z_CMB_list"][-1], res_der_fl = deriv_file_to_use, params = params)
                    the_data = helper_functions.merge_calib(the_data = the_data, dparam_dzps = dparam_dzps, current_sn_ind = current_sn_ind, uncertainties = calibration_uncertainties, check_1 = True)

                
                dparam_dzps, add_mag_electron = helper_functions.get_electron_scattering(the_data["z_CMB_list"][-1], params = params)
                the_data = helper_functions.merge_calib(the_data = the_data, dparam_dzps = dparam_dzps, current_sn_ind = current_sn_ind, uncertainties = calibration_uncertainties, check_1 = True)
                assert add_mag_electron < 0
                the_data["mB_list"][-1] += add_mag_electron

                dparam_dzps = helper_functions.get_lensing_bias(the_data["z_CMB_list"][-1], lensing_ifn)
                the_data = helper_functions.merge_calib(the_data = the_data, dparam_dzps = dparam_dzps, current_sn_ind = current_sn_ind,
                                                        uncertainties = calibration_uncertainties, check_1 = True)
                

                if params["remap_x1"] != None:
                    new_x1, x1_slope = helper_functions.remap_x1(the_data["x1_list"][-1], params)
                    the_data["x1_list"][-1] = new_x1
                    x1x1 *= x1_slope**2.
                    mBx1 *= x1_slope
                    x1c *= x1_slope


                ########################################## Peculiar Velocity Dispersion and Bulk Flows ##########################################
                

                # This formula is in Davis+ 2010, but they reference K09
                total_pec_vel_on_diag = (   params["pec_vel_disp"]*(5./log(10.))*(the_data["z_CMB_list"][-1] + 1.)/(the_data["z_CMB_list"][-1]*(1 + the_data["z_CMB_list"][-1]/2.))   )**2.
                total_bulk_quad = 0.
                
                if this_redshift_cmb < 0.1 and (params["include_pec_cov"] == 1):

                    dists = (bulk_RA - this_RA)**2. + (bulk_Dec - this_Dec)**2. + 1e6*(bulk_z - this_redshift_cmb)**2.
                    
                    bulk_inds = argsort(dists)
                    bulk_ind = bulk_inds[0]

                    assert dists[bulk_ind] < 2, "Couldn't find " + snpath.split("/")[-1] + ". You need to regenerate the bulk flow files or run with include_pec_cov set to 0."

                    for bulk_i in range(len(bulk_eig)):
                        key = "BULK_%03i" % bulk_i
                        if not the_data["calib_names"].count(key):
                            the_data["calib_names"].append(key)
                        calib_ind = the_data["calib_names"].index(key)
                        the_data["d_mBx1c_dcalib_list"][current_sn_ind, 0, calib_ind] = bulk_eig[bulk_i, bulk_ind]
                        total_bulk_quad += bulk_eig[bulk_i, bulk_ind]**2.
                        print("setting ", bulk_eig[bulk_i, bulk_ind], the_data["d_mBx1c_dcalib_list"][current_sn_ind, 0, calib_ind], "bulk_i", bulk_i, "bulk_ind", bulk_ind, "current_sn_ind", current_sn_ind)

                    key = "corr_redshift_sys"
                    if not the_data["calib_names"].count(key):
                        the_data["calib_names"].append(key)
                    calib_ind = the_data["calib_names"].index(key)
                    the_data["d_mBx1c_dcalib_list"][current_sn_ind, 0, calib_ind] = (3.3e-5)*(5./log(10.))*(the_data["z_CMB_list"][-1] + 1.)/(the_data["z_CMB_list"][-1]*(1 + the_data["z_CMB_list"][-1]/2.))

                    

                print("total_pec_vel_on_diag ", total_pec_vel_on_diag, the_data["z_CMB_list"][-1])
                total_pec_vel_on_diag -= total_bulk_quad
                total_pec_vel_on_diag = clip(total_pec_vel_on_diag, 0, 100)
                print("total_remaining to add ", total_pec_vel_on_diag, the_data["z_CMB_list"][-1])

                mBmB += total_pec_vel_on_diag
                
                ########################################## Done with Ingredients for Covariance Matrix ##########################################

                
                the_data["mBx1c_cov_list"] = concatenate((the_data["mBx1c_cov_list"], array([[[mBmB, mBx1, mBc],
                                                                                              [mBx1, x1x1, x1c],
                                                                                              [mBc, x1c, cc]]], dtype=float64) + extra_cmat   ), axis = 0)

                dparam_dzps = helper_functions.get_dparam_dzps(deriv_file_to_use, this_redshift_helio, calibration_paths = calibration_paths)
                
                the_data = helper_functions.merge_calib(the_data = the_data, dparam_dzps = dparam_dzps, current_sn_ind = current_sn_ind,
                                                        uncertainties = calibration_uncertainties)


                

                current_sn_ind += 1
            else:
                print("Skipping...", snpath, end=' ')
                for j in range(len(okay_names)):
                    if not okay_to_add[j]:
                        print(okay_names[j], end=' ')
                print()


    #assert len(the_data["calib_names"]) == the_data["d_mBx1c_dcalib_list"].shape[-1], "calib_names %i d_mBx1c_dcalib_list %s" % (len(the_data["calib_names"]), str(the_data["d_mBx1c_dcalib_list"].shape))
    
    the_data["d_mBx1c_dcalib_list"] = the_data["d_mBx1c_dcalib_list"][:len(the_data["mB_list"]), :, :len(the_data["calib_names"])]

    print('the_data["d_mBx1c_dcalib_list"].shape ', the_data["d_mBx1c_dcalib_list"].shape)
    print('the_data["calib_names"] ', the_data["calib_names"])
    
    save_img([the_data["d_mBx1c_dcalib_list"][:,0,:], the_data["d_mBx1c_dcalib_list"][:,1,:], the_data["d_mBx1c_dcalib_list"][:,2,:]], "d_mBx1c_dcalib_list.fits")

    print("read cov shape ", the_data["mBx1c_cov_list"].shape)

    for current_sample in range(len(the_data["sample_names"])):
        inds = where(the_data["sample_list"] == current_sample)
        plt.subplot(2,1,1)
        plt.errorbar(the_data["z_CMB_list"][inds], the_data["c_list"][inds], yerr = sqrt(the_data["mBx1c_cov_list"][:,2,2][inds]), fmt ='.', capsize = 0)
        plt.xlim(0., 0.1)
        plt.subplot(2,1,2)
        plt.errorbar(the_data["z_CMB_list"][inds], the_data["c_list"][inds], yerr = sqrt(the_data["mBx1c_cov_list"][:,2,2][inds]), fmt ='.', capsize = 0)

    plt.savefig("c_vs_z.pdf")
    f_read.close()

    if the_data["d_mBx1c_dz_list"] == []:
        the_data["d_mBx1c_dz_list"] = zeros([0,3], dtype=float64)

        
    return the_data




################################################# Redshifts for Integration ###################################################

def get_redshifts(redshifts):
    appended_redshifts = arange(0., 2.51, 0.1)
    tmp_redshifts = concatenate((redshifts, appended_redshifts))
    
    sort_inds = list(argsort(tmp_redshifts))
    unsort_inds = [sort_inds.index(i) for i in range(len(tmp_redshifts))]
    
    tmp_redshifts = sort(tmp_redshifts)
    redshifts_sort_fill = sort(concatenate((tmp_redshifts, 0.5*(tmp_redshifts[1:] + tmp_redshifts[:-1]))))
    
    return redshifts_sort_fill, unsort_inds, len(appended_redshifts)


################################################# Redshift Coefficients for Population ###################################################


def plot_coeffs(z_list, redshift_coeffs):
    plt.figure(2)

    sqrtn = int(np.ceil(np.sqrt(len(redshift_coeffs[0]))))

    plt.figure(figsize = (4*sqrtn, 3*sqrtn))
    
    for j in range(len(redshift_coeffs[0])):
        plt.subplot(sqrtn, sqrtn, j+1)
        plt.plot(z_list, redshift_coeffs[:,j], '.')

    plt.savefig("redshift_coeffs.pdf")
    plt.close()

    #assert np.min(np.max(redshift_coeffs, axis = 1) - np.min(redshift_coeffs, axis = 1)) > 0

def get_redshift_coeffs(z_list, p_high_mass, separate_mass_x1c, redshift_coeff_type):
    """redshift_coeff_type could be ("a", 1) or ("a", 3) for a population that varies with a(t)
    redshift_coeff_type could be ("sample", [0.0, 0.4, 1.0]) for a population that is allowed to be different low-z, mid-z, high-z"""

    z_list = np.array(z_list)
    set_list = np.array(the_data["sample_list"])

    if redshift_coeff_type[1].count("."):
        n_z = len(redshift_coeff_type[1:])
    else:
        n_z = int(redshift_coeff_type[1])
    
    actual_n_x1c_star = n_z*(1 + separate_mass_x1c)

    redshift_coeffs = np.zeros([len(z_list), actual_n_x1c_star], dtype=np.float64)

    if n_z == 1:
        if separate_mass_x1c:
            redshift_coeffs[:,0] = p_high_mass
            redshift_coeffs[:,1] = 1 - p_high_mass
        else:
            redshift_coeffs += 1

        plot_coeffs(z_list, redshift_coeffs)
        return redshift_coeffs

    if redshift_coeff_type[0] == "a":
        a_list = 1./(1. + np.array(z_list))
        a_nodes = np.linspace(min(a_list) - 1e-5, max(a_list) + 1e-5, n_z)
    
        for i in range(len(z_list)):
            for j in range(n_z):
                coeffs = zeros(n_z, dtype=float64)
                coeffs[j] = 1

                ifn = interp1d(a_nodes, coeffs, kind = 'linear')

                if separate_mass_x1c:
                    redshift_coeffs[i,j] = ifn(a_list[i])*p_high_mass[i]
                    redshift_coeffs[i,n_z + j] = ifn(a_list[i])*(1. - p_high_mass[i])
                else:
                    redshift_coeffs[i,j] = ifn(a_list[i])


    elif redshift_coeff_type[0] == "sample":
        zs_to_match = np.array([float(item) for item in redshift_coeff_type[1:]])
        print("zs_to_match", zs_to_match)
        
        for set_ind in np.unique(set_list):
            mean_z = np.mean(z_list[np.where(set_list == set_ind)])
            
            j = np.argmin(np.abs(zs_to_match - mean_z))
            print("set_ind", set_ind, "mean_z", mean_z, "j", j)

            if separate_mass_x1c:
                redshift_coeffs[:,j] += (set_list == set_ind)*p_high_mass
                redshift_coeffs[:,n_z + j] += (set_list == set_ind)*(1. - p_high_mass)
            else:
                redshift_coeffs[:,j] += (set_list == set_ind)*1.
    else:
        assert 0, "Unknown redshift_coeff_type " + str(redshift_coeff_type)
    
    plot_coeffs(z_list, redshift_coeffs)
    return redshift_coeffs

################################################# Binned mu ###################################################


def zcount(z, zmin, zmax):
    return sum((array(z) >= zmin)*(array(z) < zmax))

"""
def zbins_chi2fn(P, alldata):
    z = alldata[0]

    dbin = P[1:] - P[:-1]
    if dbin.min() < 0.02:
        return 1e100
    if dbin.max() > 0.2:
        return 1e100

    zmean = 0.5*(P[1:] + P[:-1])

    should_be_const = dbin/(1. + zmean)
    should_be_const -= np.mean(should_be_const)

    chi2 = np.dot(should_be_const, should_be_const)

    
    for i in range(len(P) - 1):
        nsne = zcount(z, P[i], P[i+1])
        chi2 += (nsne < 10.)*(nsne - 10.)**10.
        chi2 += (nsne > 100.)*((nsne - 100.)/20.)**2.

    return chi2
"""

"""
def get_equal_a_bins(z, n_to_add):
    if z.max()/z.min() < 20:
        ministart = np.linspace(0, 1, 4)
        assert cosmo_model != 2
    else:
        ministart = np.linspace(0, 1, 30 - n_to_add)

    if n_to_add == 0:
        zbins = -1. + (   1 + z.max() + 0.002   )**ministart * (   1 + z.min() - 0.001   )**(1. - ministart)
        return zbins
    else:
        highest_X = np.sort(stan_data["redshifts"])

        zbins = -1. + (   1 + highest_X[-10*n_to_add - 1]   )**ministart * (   1 + z.min() - 0.001   )**(1. - ministart)

        for i in range(n_to_add):
            zbins = np.append(zbins, highest_X[-10*n_to_add + (i + 1)*10 - 1] + 0.002)
        zbins[-1] += 0.001
        
        return zbins
"""

def add_zbins(stan_data, cosmo_model):
    # For binned mu
    
    stan_data["cosmo_model"] = cosmo_model

    print("min, max", stan_data["redshifts"].min(), stan_data["redshifts"].max())
    if stan_data["redshifts"].min() == stan_data["redshifts"].max() or ([2, 6].count(cosmo_model) == 0):
        stan_data["zbins"] = [stan_data["redshifts"][0]]
        stan_data["n_zbins"] = 1
        stan_data["dmu_dbin"] = ones([stan_data["n_sne"], stan_data["n_zbins"]], dtype=float64)
        stan_data["dmudz_dbin"] = zeros([stan_data["n_sne"], stan_data["n_zbins"]], dtype=float64)
        stan_data["mu_const"] = np.zeros(stan_data["n_sne"], dtype=np.float64)
        
        return stan_data


    #if cosmo_model == 2:
    #    zbins = np.exp(np.linspace(np.log(stan_data["redshifts"].min()*0.999),
    #                               np.log(stan_data["redshifts"].max()*1.001), 30))
    #else:
    assert cosmo_model == 6 or cosmo_model == 2
    zsort = np.sort(stan_data["redshifts"])

    print("zsort", zsort[-10:])

    zbins = [zsort[-1]*1.001]
    step = 10
    minstepsize = 0.1
    min_sn_bin = 10
    ind = -1 - min_sn_bin
    z_cutoff_for_05 = 0.8

    while step > minstepsize:
        step = zbins[0] - zsort[ind]
        minstepsize = ((zbins[0] + zsort[ind])*0.5 > z_cutoff_for_05)*0.05 + 0.05

        if step > minstepsize:
            zbins = [zsort[ind]] + zbins
            ind -= min_sn_bin

    print("zbins high z", zbins)


    zbins = np.concatenate((
        np.linspace(0.05, z_cutoff_for_05, int(np.around(z_cutoff_for_05/0.05))),
        np.linspace(z_cutoff_for_05, zbins[0], int(np.around((zbins[0] - z_cutoff_for_05)/0.1)) + 1)[1:-1],
        zbins))



    zbins = np.array(zbins)


    print("zbins", zbins, list(zbins))


    stan_data["zbins"] = zbins
    
    """
    stan_data["zbins"] = [0.99999*stan_data["redshifts"].min()]

    while max(stan_data["zbins"]) < max(stan_data["redshifts"]):
        #zstep = 0.125
        zstep = 0.02*(1. + stan_data["zbins"][-1])
        while (zcount(stan_data["redshifts"], stan_data["zbins"][-1], stan_data["zbins"][-1] + zstep) < 10.) and (stan_data["zbins"][-1] + zstep < stan_data["redshifts"].max()):
            zstep *= 1.5

        #stan_data["zbins"].append(stan_data["zbins"][-1]*exp(zstep) + 0.001)
        stan_data["zbins"].append(stan_data["zbins"][-1] + zstep)
    """


    stan_data["n_zbins"] = len(stan_data["zbins"])

    f = open("zbins.txt", 'w')
    for zbin in stan_data["zbins"]:
        f.write(str(zbin) + '\n')
    f.close()

    plt.figure()
    plt.hist(stan_data["redshifts"], bins = 20)
    plt.plot(stan_data["zbins"], [100]*stan_data["n_zbins"], '.', color = 'k')
    plt.yscale('log')
    plt.savefig("redshift_binning.pdf")
    plt.close()

    stan_data["dmu_dbin"] = zeros([stan_data["n_sne"], stan_data["n_zbins"]], dtype=float64)
    stan_data["dmudz_dbin"] = zeros([stan_data["n_sne"], stan_data["n_zbins"]], dtype=float64)

    for j in range(stan_data["n_zbins"]):
        nodes = zeros(stan_data["n_zbins"], dtype=float64)
        nodes[j] = 1.

        if cosmo_model == 6:

            minz = min(stan_data["redshifts"])*0.999
            ifn = interp1d(
                np.concatenate(([0, minz], stan_data["zbins"])),
                np.concatenate(([0, minz], nodes)), kind = 'cubic')
        else:
            assert cosmo_model == 2
            ifn = interp1d(np.concatenate(([0], stan_data["zbins"])),
                           np.concatenate(([0], nodes)), kind = 'quadratic')

            
        for i in range(stan_data["n_sne"]):
            stan_data["dmu_dbin"][i, j] = ifn(stan_data["redshifts"][i])
            stan_data["dmudz_dbin"][i, j] = (ifn(stan_data["redshifts"][i] + 0.001) - ifn(stan_data["redshifts"][i]))/0.001

    if cosmo_model == 6:
        stan_data["mu_const"] = np.zeros(stan_data["n_sne"], dtype=np.float64)
    else:
        stan_data["mu_const"] = get_mu(z_list = stan_data["redshifts"],
                                       cosmo = dict(model = "flatLCDM", O_m = 0.3, O_k = 0.0, h = 0.7),
                                       z_helio_list = stan_data["zhelio"])


    plt.figure()
    plt.imshow(stan_data["dmu_dbin"])
    plt.savefig("dmu_dbin.pdf")
    plt.close()

    save_img(stan_data["dmu_dbin"], "dmu_dbin.fits")
    save_img(stan_data["dmudz_dbin"], "dmudz_dbin.fits")

    return stan_data


################################################# Init FN ###################################################

def init_fn():
    n_sne = len(the_data["x1_list"])
    n_samples = len(the_data["sample_names"])
    print("n_sne ", n_sne)
    print("n_samples ", n_samples)

    if stan_data["cosmo_model"] == 2 or stan_data["cosmo_model"] == 6:
        zbins_tmp = np.array(stan_data["zbins"])
        mu_init = 43.2 + 5*np.log10((zbins_tmp - 0.225*zbins_tmp**2.)*(1. + zbins_tmp))
    #elif stan_data["cosmo_model"] == 6:
    #    zbins_tmp = np.array(stan_data["zbins"])
    #    mu_init = zbins_tmp - 0.225*zbins_tmp**2.
    else:
        mu_init = np.zeros(stan_data["n_zbins"], dtype=np.float64)
        
            
    return {"MB": random.random(size = [(n_samples - 1)*stan_data["MB_by_sample"] + 1])*0.2 - 19.1,
            "Om": 0.3,
            "wDE": -1.01,
            "mu_zbins": mu_init,
            "alpha_angle": arctan(random.random()*0.2),
            "beta_angle_blue": arctan(random.random()*0.5 + 2.5),
            "beta_angle_red_low": arctan(random.random()*0.5 + 2.5),
            "beta_angle_red_high": arctan(random.random()*0.5 + 2.5),
            #"log10_sigma_int": log10(random.random(size = n_samples)*0.1 + 0.1),
            "mBx1c_int_variance": [0.9, 0.05, 0.05],
            #"mass_0": 10,
            "delta_0": random.random()*0.05,
            "delta_h": 0.5,
            "calibs": random.normal(size = len(the_data["calib_names"]))*0.01,
            #"blind_values": [0.]*n_samples,
            
            "true_cB": random.random(size = n_sne)*0.02 - 0.01 + clip(the_data["c_list"]/2., -0.2, 1.0),
            "true_cR_unit": random.random(size = n_sne)*0.5 + 0.5, #random.random(size = n_sne)*0.01 + clip(the_data["c_list"]/2., 0, 1.0),
            "true_x1": random.random(size = n_sne)*0.2 - 0.1 + the_data["x1_list"],

            "x1_star": random.random(size = stan_data["n_x1c_star"])*0.5,
            "tau_x1": -random.random(size = stan_data["n_x1c_star"]),
            "R_x1": random.random(size = stan_data["n_x1c_star"])*0.5 + 0.25,

            "c_star": -random.random(size = stan_data["n_x1c_star"])*0.05,
            "tau_c": random.random(size = stan_data["n_x1c_star"])*0.05,
            "R_c": random.random(size = stan_data["n_x1c_star"])*0.05 + 0.02,
            
            "outl_frac": random.random()*0.02 + 0.01,
            "mobs_cuts": stan_data["est_mobs_cuts"] + random.normal(size = n_samples)*0.1, "mobs_cut_sigmas": [0.5]*n_samples,

            "dz": random.normal(size = stan_data["n_photoz"])*0.01
        }
            
            

################################################# Main Program ###################################################

inputfl = sys.argv[1]
print("cosmo_model: 1 for Om, 2 for binned mu, 3 for Omega_m-w, 4 for q0-j0, 5 for Omega_m-w0-wa, 6 for binned mu with comoving interpolation")
cosmo_model = int(sys.argv[2])



if inputfl.count("pickle"):
    (the_data, stan_data, params) = pickle.load(gzip.open(inputfl, "rb"))
else:
    params = helper_functions.get_params(inputfl)

    ################################################# And Go! ###################################################
    assert params["iter"] % 4 == 0, "iter should be a multiple of four! "  + str(params["iter"])

    the_data = read_data(params)
    names_of_all_inputs = [item.split(".")[0].split("/")[-1].replace("_v1", "") for item in params["filenamelist"]]
    if len("".join(names_of_all_inputs)) > 200:
        names_of_all_inputs = [item[:5] for item in names_of_all_inputs]
        
    samples_txt = "_".join(names_of_all_inputs)


    for i, sample in enumerate(the_data["sample_names"]):
        print(sample, sum(the_data["sample_list"] == i))


    n_sne = len(the_data["c_list"])

    obs_mBx1c = []
    obs_mBx1c_cov = zeros((3*n_sne, 3*n_sne), dtype=float64)

    for i in range(n_sne):
        obs_mBx1c.append([the_data["mB_list"][i], the_data["x1_list"][i], the_data["c_list"][i]])

    obs_mBx1c_cov = the_data["mBx1c_cov_list"]

    redshifts_sort_fill, unsort_inds, nzadd = get_redshifts(the_data["z_CMB_list"])

    p_high_mass = 0.5*(1. + erf((np.array(the_data["mass"]) - 10.)/(np.sqrt(2.) * np.array(the_data["mass_err"]))))

    redshift_coeffs = get_redshift_coeffs(z_list = the_data["z_CMB_list"],
                                          p_high_mass = p_high_mass,
                                          separate_mass_x1c = params["separate_mass_x1c"],
                                          redshift_coeff_type = params["redshift_coeff_type"])
    

    BAOCMB_Om_w0_wa_mean, BAOCMB_Om_w0_wa_covmatrix = pickle.load(open(os.environ["UNITY"] + "/other_cosmology/BAOCMB_Omw0wa.pickle", 'rb'))

    if params["fix_Om"] > 0:
        print("fix_Om, so turning off BAOCMB cov mat!")
        BAOCMB_Om_w0_wa_mean = [0.3, -1, 0.]
        BAOCMB_Om_w0_wa_covmatrix = np.diag([100, 100., 100.])
    
    stan_data = {"n_sne": n_sne, "nzadd": nzadd,
                 "n_samples": len(the_data["sample_names"]),
                 "redshift_coeffs": redshift_coeffs,
                 "n_calib": len(the_data["calib_names"]),
                 "d_mBx1c_d_calib": the_data["d_mBx1c_dcalib_list"],
                 "n_x1c_star": len(redshift_coeffs[0]), # 3 = 3 scale-factor nodes
                 "threeD_unexplained": params["threeD_unexplained"],
                 "mass": the_data["mass"],
                 "mass_err": the_data["mass_err"],
                 "p_high_mass": p_high_mass,
                 "do_host_mass": params["do_host_mass"], "fix_Om": params["fix_Om"], "MB_by_sample": params["MB_by_sample"], 
                 # The +1 here is for Stan's indexing, which is from 1 not 0
                 "sample_list": the_data["sample_list"] + 1,
                 "zhelio": the_data["z_helio_list"],
                 "redshifts": the_data["z_CMB_list"],
                 "redshifts_sort_fill": redshifts_sort_fill, "unsort_inds": unsort_inds,
                 "obs_mBx1c": array(obs_mBx1c),
                 "obs_mBx1c_cov": array(obs_mBx1c_cov),
                 "do_blind": params["do_blind"],
                 "do_twoalphabeta": params["do_twoalphabeta"],

                 "outl_frac_prior_lnmean": log(params["outl_frac"]),
                 "outl_frac_prior_lnwidth": 0.5,

                 "n_photoz": the_data["n_photoz"],
                 "d_mBx1c_dz_list": the_data["d_mBx1c_dz_list"],
                 "photo_z0": the_data["photo_z0"],
                 "photo_dz": the_data["photo_dz"],
                 "spike_redshift_prob": [0.8]*the_data["n_photoz"],
                 "photoz_inds": the_data["photoz_inds"],
                 "photo_spikez": the_data["photo_spikez"],


                 "est_mobs_cuts": the_data["est_mobs_cuts"],
                 "est_mobs_sigmas": the_data["est_mobs_sigmas"],
                 "mobs_cut0": the_data["mobs_cut0"], "mobs_cut1": the_data["mobs_cut1"],
                 "BAOCMB_Om_w0_wa_mean": BAOCMB_Om_w0_wa_mean, "BAOCMB_Om_w0_wa_covmatrix": BAOCMB_Om_w0_wa_covmatrix
             }

    plt.figure()
    plt.plot(stan_data["redshifts"], stan_data["mobs_cut0"], '.')
    plt.plot(stan_data["redshifts"], stan_data["mobs_cut1"], '.')
    plt.savefig("k_corr_check.pdf")
    plt.close()

    pickle.dump((the_data, stan_data, params), gzip.open("inputs_" + samples_txt + ".pickle", "wb"))


stan_data = add_zbins(stan_data, cosmo_model)

print("nzadd ", stan_data['nzadd'])
# print stan_data['n_sne']
# print stan_data['n_samples']
# print stan_data['sample_list'].shape
# print stan_data['redshift'].shape
# print stan_data['obs_mBx1c']
# print stan_data['obs_mBx1c_cov'].shape



    
print("Running...")

smpfl = os.environ["UNITY"] + "/scripts/stan_code_" + os.uname()[1] + ".pickle"
smfl = params["stan_code"].replace("$UNITY", os.environ["UNITY"])
print(("smpfl", smpfl))

f = open(smfl, 'r')
smfl_lines = f.read()
f.close()

try:
    print("Trying to load")
    sm, sc = pickle.load(open(smpfl, 'rb'))
    if sc != smfl_lines:
        print("Okay. Need to recompile!")
        raise_time

    fit = sm.sampling(data=stan_data,
                      iter=10, chains=1, n_jobs = 1, refresh = 1, init = init_fn, sample_file = params["sample_file"])
        
except:
    sm = pystan.StanModel(file=smfl)
    pickle.dump((sm, smfl_lines), open(smpfl, 'wb'))


fit = sm.sampling(data=stan_data,
                  iter=params["iter"], chains=params["chains"], n_jobs = params["n_jobs"], refresh = 10, init = init_fn, sample_file = params["sample_file"])
                  # pars = ["beta", "dbeta", "alpha", "dalpha", "MB", "Om", "sigma_int", "x1_star", "R_x1", "c_star", "R_c", "calibs"])#, sample_file = "/Users/rubind/Dropbox/samples.txt")


fit_params = fit.extract(permuted = True)


try:
    fit_params = filter_fit_params(fit_params, "MB", params["chains"], params["iter"]/2) # burns the first half of the chain, so iter/2
except:
    print("Couldn't filter bad chains! One or more chains may be bad!")

#summarize_parameters(fit_params)



try:
    mu_cov = np.cov(fit_params["mu_zbins"].T)
    whole_mat = np.zeros([len(mu_cov) + 1]*2, dtype=np.float64)
    whole_mat[1:, 1:] = np.linalg.inv(mu_cov)
    whole_mat[1:, 0] = np.median(fit_params["mu_zbins"], axis = 0)
    whole_mat[0, 1:] = stan_data["zbins"]
    
    save_img(whole_mat, "mu_mat.fits")
except:
    print("Couldn't save whole_mat")


del_keys = []
for key in fit_params:
    sh = np.array(fit_params[key].shape)

    if np.any(sh[1:] > params["max_params_to_save"]):
        print(key, " is too big to save!", sh)
        del_keys.append(key)

print("del_keys", del_keys)
for key in del_keys:
    del fit_params[key]

    
try:
    samples_txt
    pickle.dump(fit_params, gzip.open("samples_" + samples_txt + ".pickle", "wb"))
except:
    pickle.dump(fit_params, gzip.open("samples.pickle", "wb"))

    

print("I hope you have a log file:")

try:
    print(fit.stansummary(digits_summary=5))
except:
    print("Couldn't print fit! Something is very wrong!")


