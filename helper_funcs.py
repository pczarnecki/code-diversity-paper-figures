## ARTS as the line-by-line code
import os
# set global variables BEFORE importing ARTS
os.environ['ARTS_DATA_PATH'] = '../arts-cat-data-2.6.14:../arts-xml-data-2.6.14'

import pyarts
from pyarts.arts import convert
# the fluxsim module
import FluxSimulator as fsm


# helper packages
import numpy as np
import xarray as xr
from scipy.constants import speed_of_light as c
import matplotlib.pyplot as plt

import typhon


def concat_arts_output_batch(file_dir, file_superscript, file_subscript, regime, allsky = True):

    if regime == 'sw': 
        d1, d2, d3, d4, d5 = 0.002, 0.001, 0.002, 0.02, 1
        m = 100
        ## set wavenumbers at fine spectral discretization
        wavenumbers_sw = np.concatenate([np.arange(250, 2200, m*d1), 
                        np.arange(2200, 2400, m*d2),
                        np.arange(2400, 5150, m*d3),
                        np.arange(5150, 16000, m*d4),
                        #np.arange(16000, 35000, d5)])
                        np.arange(16000, 48979, d5),
                        np.arange(48979, 49950, d5)])
        data = np.load(file_dir + file_superscript + '989405049137400.0_1049243623754200.0_' + file_subscript + '.npy', allow_pickle=True).item()
        array_of_irradiance_field_clearsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))
        array_of_irradiance_field_allsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))

        for i in np.arange(0, 33):
            frequencies_sw = convert.kaycm2freq(wavenumbers_sw)
            if i == 32:
                frequencies_sw = frequencies_sw[i*1997:]

            else:
                frequencies_sw = frequencies_sw[i*1997:i*1997+1997]
            #dnu = np.max(frequencies_sw) - np.min(frequencies_sw)
            #print(file_dir + file_superscript + str(np.min(frequencies_sw)) + '_' + str(np.max(frequencies_sw)) + '_' + file_subscript + '.npy')
            data = np.load(file_dir + file_superscript + str(np.min(frequencies_sw)) + '_' + str(np.max(frequencies_sw)) + '_' + file_subscript + '.npy', allow_pickle=True).item()
            if allsky == True:
                array_of_irradiance_field_allsky += np.array(data["array_of_irradiance_field_allsky"])
            array_of_irradiance_field_clearsky += np.array(data["array_of_irradiance_field_clearsky"])

        results = xr.Dataset(
            data_vars = dict(
                flux_cs = (["index", "pressure", "direction"], array_of_irradiance_field_clearsky),
                flux_as = (["index", "pressure", "direction"], array_of_irradiance_field_allsky),
            ),
            coords = dict(
                index = np.array(data["array_of_index"]),
                pressure = np.array(data["array_of_pressure"])[0],
                direction = np.array(["down", "up"])
            ),
        )

    elif regime == 'lw':
        d1, d2, d3 = 0.0002, 0.001, 0.005
        m = 150
        ## set wavenumbers at fine spectral discretization
        wavenumbers_lw = np.concatenate([np.arange(5, 1300, m*d1), 
                        np.arange(1300, 1700, m*d2),
                        np.arange(1700, 3257, m*d3)])

        data = np.load(file_dir + file_superscript + '1586201895278.0117_3021608184182.0234_' + file_subscript + '.npy', allow_pickle=True).item()
        array_of_irradiance_field_clearsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))
        array_of_irradiance_field_allsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))


        for i in np.arange(0, 29):
            frequencies_lw = convert.kaycm2freq(wavenumbers_lw)
            frequencies_lw = frequencies_lw[i*1597:i*1597+1597]
            #dnu = np.max(frequencies_sw) - np.min(frequencies_sw)
            data = np.load(file_dir + file_superscript + str(np.min(frequencies_lw)) + '_' + str(np.max(frequencies_lw)) + '_'  + file_subscript + '.npy', allow_pickle=True).item()
            if allsky == True:
                if 'array_of_irradiance_field_allsky' in data.keys():
                    array_of_irradiance_field_allsky += np.array(data["array_of_irradiance_field_allsky"])
                else:
                    array_of_irradiance_field_allsky += np.array(data["array_of_irradiance_field_clearsky"])

            array_of_irradiance_field_clearsky += np.array(data["array_of_irradiance_field_clearsky"])

        results = xr.Dataset(
            data_vars = dict(
                flux_cs = (["index", "pressure", "direction"], array_of_irradiance_field_clearsky),
                flux_as = (["index", "pressure", "direction"], array_of_irradiance_field_allsky),
            ),
            coords = dict(
                index = np.array(data["array_of_index"]),
                pressure = np.array(data["array_of_pressure"])[0],
                direction = np.array(["down", "up"])
            ),

        )
    
    else:
        print("regime must be either sw or lw")

    return results


def concat_arts_output_batch_liq(file_dir, file_superscript, file_subscript, regime, cs_filename, allsky = True):

    if regime == 'sw': 
        d1, d2, d3, d4, d5 = 0.002, 0.001, 0.002, 0.02, 1
        m = 100
        ## set wavenumbers at fine spectral discretization
        wavenumbers_sw = np.concatenate([np.arange(250, 2200, m*d1), 
                        np.arange(2200, 2400, m*d2),
                        np.arange(2400, 5150, m*d3),
                        np.arange(5150, 16000, m*d4),
                        #np.arange(16000, 35000, d5)])
                        np.arange(16000, 48979, d5),
                        np.arange(48979, 49950, d5)])
        data = np.load(file_dir + file_superscript + '989405049137400.0_1049243623754200.0_' + file_subscript + '.npy', allow_pickle=True).item()
        array_of_irradiance_field_clearsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))
        array_of_irradiance_field_allsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))

        for i in np.arange(0, 33):
            frequencies_sw = convert.kaycm2freq(wavenumbers_sw)
            if i == 32:
                frequencies_sw = frequencies_sw[i*1997:]

            else:
                frequencies_sw = frequencies_sw[i*1997:i*1997+1997]
            #dnu = np.max(frequencies_sw) - np.min(frequencies_sw)
            #print(file_dir + file_superscript + str(np.min(frequencies_sw)) + '_' + str(np.max(frequencies_sw)) + '_' + file_subscript + '.npy')
            data = np.load(file_dir + file_superscript + str(np.min(frequencies_sw)) + '_' + str(np.max(frequencies_sw)) + '_' + file_subscript + '.npy', allow_pickle=True).item()
            data_cs = np.load(file_dir + file_superscript + str(np.min(frequencies_sw)) + '_' + str(np.max(frequencies_sw)) + '_' + cs_filename + '_cs.npy', allow_pickle=True).item()
            if allsky == True:
                array_of_irradiance_field_allsky += np.array(data["array_of_irradiance_field_allsky"])
            array_of_irradiance_field_clearsky += np.array(data_cs["array_of_irradiance_field_allsky"])

        results = xr.Dataset(
            data_vars = dict(
                flux_cs = (["index", "pressure", "direction"], array_of_irradiance_field_clearsky),
                flux_as = (["index", "pressure", "direction"], array_of_irradiance_field_allsky),
            ),
            coords = dict(
                index = np.array(data["array_of_index"]),
                pressure = np.array(data["array_of_pressure"])[0],
                direction = np.array(["down", "up"])
            ),
        )

    elif regime == 'lw':
        d1, d2, d3 = 0.0002, 0.001, 0.005
        m = 150
        ## set wavenumbers at fine spectral discretization
        wavenumbers_lw = np.concatenate([np.arange(5, 1300, m*d1), 
                        np.arange(1300, 1700, m*d2),
                        np.arange(1700, 3257, m*d3)])

        data = np.load(file_dir + file_superscript + '1586201895278.0117_3021608184182.0234_' + file_subscript + '.npy', allow_pickle=True).item()
        array_of_irradiance_field_clearsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))
        array_of_irradiance_field_allsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))


        for i in np.arange(0, 29):
            frequencies_lw = convert.kaycm2freq(wavenumbers_lw)
            frequencies_lw = frequencies_lw[i*1597:i*1597+1597]
            #dnu = np.max(frequencies_sw) - np.min(frequencies_sw)
            data = np.load(file_dir + file_superscript + str(np.min(frequencies_lw)) + '_' + str(np.max(frequencies_lw)) + '_'  + file_subscript + '.npy', allow_pickle=True).item()
            data_cs = np.load(file_dir + file_superscript + str(np.min(frequencies_lw)) + '_' + str(np.max(frequencies_lw)) + '_' + cs_filename + '_cs.npy', allow_pickle=True).item()
            if allsky == True:
                if 'array_of_irradiance_field_allsky' in data.keys():
                    array_of_irradiance_field_allsky += np.array(data["array_of_irradiance_field_allsky"])
                    array_of_irradiance_field_clearsky += np.array(data_cs["array_of_irradiance_field_allsky"])

                else:
                    array_of_irradiance_field_allsky += np.array(data["array_of_irradiance_field_clearsky"])
                    array_of_irradiance_field_clearsky += np.array(data_cs["array_of_irradiance_field_clearsky"])


        results = xr.Dataset(
            data_vars = dict(
                flux_cs = (["index", "pressure", "direction"], array_of_irradiance_field_clearsky),
                flux_as = (["index", "pressure", "direction"], array_of_irradiance_field_allsky),
            ),
            coords = dict(
                index = np.array(data["array_of_index"]),
                pressure = np.array(data["array_of_pressure"])[0],
                direction = np.array(["down", "up"])
            ),

        )
    
    else:
        print("regime must be either sw or lw")

    return results



def stdout_to_dephy_clr(path_in, path_out):
    ## clear-sky + contrail only 

    ## default LMDZ vmrs
    # from 1D/INPUT/DEF/config.def
    CO2_default = 0.36886500E+03 * 1e-06
    CH4_default = 1.65e-06
    N2O_default = 306.e-09
    CFC11_default = 280.e-12
    CFC12_default = 484.e-12

    combine = xr.open_mfdataset([path_in + 'histins.nc'], compat = 'override')
    combine = combine.rename({'zfull':'zf', 'zhalf':'zh'})
    pressure = combine["presnivs"].broadcast_like(combine["zf"])
    combine = combine.assign({'pf':pressure})

    combine = combine.assign({'o3':(('time_counter', 'presnivs', 'lat', 'lon'), combine.ozone.data * (48/28.97))})

    ## if no gases, add defaults
    if 'co2' not in combine.keys():
        combine = combine.assign({'co2': (('time_counter', 'presnivs', 'lat', 'lon'), CO2_default * np.ones_like(combine.zf))}) 

    if 'ch4' not in combine.keys():
        combine = combine.assign({'ch4': (('time_counter', 'presnivs', 'lat', 'lon'), CH4_default * np.ones_like(combine.zf))}) 

    if 'n2o' not in combine.keys():
        combine = combine.assign({'n2o': (('time_counter', 'presnivs', 'lat', 'lon'), N2O_default * np.ones_like(combine.zf))}) 

    if 'cfc11' not in combine.keys():
        combine = combine.assign({'cfc11': (('time_counter', 'presnivs', 'lat', 'lon'), CFC11_default * np.ones_like(combine.zf))}) 

    if 'cfc12' not in combine.keys():
        combine = combine.assign({'cfc12': (('time_counter', 'presnivs', 'lat', 'lon'), CFC12_default * np.ones_like(combine.zf))}) 

    combine = combine.drop_vars('zh')
    combine = combine.drop_vars({'re', 'rneb'})
    combine = combine.rename({'ovap':'qv', 'qi_cont':'qi', 'tsol':'tsurf', 'ref_ice_cont':'ri', 'cfcseri':'rneb'})
    
    combine.to_netcdf(path_out)    


def compute_cre(ecrad_data = None, arts_data_sw = None, arts_data_lw = None):
    LW_ecrad, SW_ecrad, SW_ARTS, LW_ARTS = None, None, None, None

    if ecrad_data != None:
        LW_ecrad = ((ecrad_data.flux_up_lw_clear - ecrad_data.flux_dn_lw_clear).isel(half_level = 0)
                            - (ecrad_data.flux_up_lw - ecrad_data.flux_dn_lw).isel(half_level = 0))
        SW_ecrad = ((ecrad_data.flux_up_sw_clear - ecrad_data.flux_dn_sw_clear).isel(half_level = 0) - 
                                (ecrad_data.flux_up_sw - ecrad_data.flux_dn_sw).isel(half_level = 0))

    if arts_data_sw != None:
        SW_ARTS = ((arts_data_sw.flux_cs.sel(direction = 'up') - arts_data_sw.flux_cs.sel(direction = 'down')).isel(pressure = -1) - 
                            (arts_data_sw.flux_as.sel(direction = 'up') - arts_data_sw.flux_as.sel(direction = 'down')).isel(pressure = -1))

    if arts_data_lw != None:
        LW_ARTS = ((arts_data_lw.flux_cs.sel(direction = 'up') - arts_data_lw.flux_cs.sel(direction = 'down')).isel(pressure = -1) - 
                                (arts_data_lw.flux_as.sel(direction = 'up') - arts_data_lw.flux_as.sel(direction = 'down')).isel(pressure = -1))

    return LW_ecrad, SW_ecrad, LW_ARTS, SW_ARTS

def stdout_to_dephy_clr_hl(path_in, path_out):
    ## clear-sky + contrail only 

    ## default LMDZ vmrs
    # from 1D/INPUT/DEF/config.def
    CO2_default = 0.36886500E+03 * 1e-06
    CH4_default = 1.65e-06
    N2O_default = 306.e-09
    CFC11_default = 280.e-12
    CFC12_default = 484.e-12

    combine = xr.open_mfdataset([path_in + 'histins.nc'], compat = 'override')
    combine = combine.rename({'zfull':'zf', 'zhalf':'zh'})
    pressure = combine["presnivs"].broadcast_like(combine["zh"])
    combine = combine.assign({'pf':pressure})

    combine = combine.assign({'o3':(('time_counter', 'presnivs', 'lat', 'lon'), combine.ozone.data * (48/28.97))})

    ## if no gases, add defaults
    if 'co2' not in combine.keys():
        combine = combine.assign({'co2': (('time_counter', 'presnivs', 'lat', 'lon'), CO2_default * np.ones_like(combine.zh))}) 

    if 'ch4' not in combine.keys():
        combine = combine.assign({'ch4': (('time_counter', 'presnivs', 'lat', 'lon'), CH4_default * np.ones_like(combine.zh))}) 

    if 'n2o' not in combine.keys():
        combine = combine.assign({'n2o': (('time_counter', 'presnivs', 'lat', 'lon'), N2O_default * np.ones_like(combine.zh))}) 

    if 'cfc11' not in combine.keys():
        combine = combine.assign({'cfc11': (('time_counter', 'presnivs', 'lat', 'lon'), CFC11_default * np.ones_like(combine.zh))}) 

    if 'cfc12' not in combine.keys():
        combine = combine.assign({'cfc12': (('time_counter', 'presnivs', 'lat', 'lon'), CFC12_default * np.ones_like(combine.zh))}) 

    combine = combine.drop_vars('zf')
    combine = combine.drop_vars({'re', 'rneb'})
    combine = combine.rename({'ovap':'qv', 'qi_cont':'qi', 'tsol':'tsurf', 'ref_ice_cont':'ri', 'cfcseri':'rneb'})
    
    combine.to_netcdf(path_out)    


def check_outputs(file_dir, file_superscript, file_subscript, regime, allsky = True):

    if regime == 'sw': 
        d1, d2, d3, d4, d5 = 0.002, 0.001, 0.002, 0.02, 1
        m = 100
        ## set wavenumbers at fine spectral discretization
        wavenumbers_sw = np.concatenate([np.arange(250, 2200, m*d1), 
                        np.arange(2200, 2400, m*d2),
                        np.arange(2400, 5150, m*d3),
                        np.arange(5150, 16000, m*d4),
                        #np.arange(16000, 35000, d5)])
                        np.arange(16000, 48979, d5),
                        np.arange(48979, 49950, d5)])
        data = np.load(file_dir + file_superscript + '989405049137400.0_1049243623754200.0_' + file_subscript + '.npy', allow_pickle=True).item()
        array_of_irradiance_field_clearsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))
        array_of_irradiance_field_allsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))

        if (np.array(data["array_of_irradiance_field_clearsky"]).shape)[0] != 10:
            print(file_dir + file_superscript + '989405049137400.0_1049243623754200.0_' + file_subscript + '.npy')

        for i in np.arange(0, 33):
            frequencies_sw = convert.kaycm2freq(wavenumbers_sw)
            if i == 32:
                frequencies_sw = frequencies_sw[i*1997:]

            else:
                frequencies_sw = frequencies_sw[i*1997:i*1997+1997]
            #dnu = np.max(frequencies_sw) - np.min(frequencies_sw)
            data = np.load(file_dir + file_superscript + str(np.min(frequencies_sw)) + '_' + str(np.max(frequencies_sw)) + '_' + file_subscript + '.npy', allow_pickle=True).item()
            if (np.array(data["array_of_irradiance_field_clearsky"]).shape)[0] != 10:
                print(file_dir + file_superscript + str(np.min(frequencies_sw)) + '_' + str(np.max(frequencies_sw)) + '_' + file_subscript + '.npy')

    elif regime == 'lw':
        d1, d2, d3 = 0.0002, 0.001, 0.005
        m = 150
        ## set wavenumbers at fine spectral discretization
        wavenumbers_lw = np.concatenate([np.arange(5, 1300, m*d1), 
                        np.arange(1300, 1700, m*d2),
                        np.arange(1700, 3257, m*d3)])

        data = np.load(file_dir + file_superscript + '1586201895278.0117_3021608184182.0234_' + file_subscript + '.npy', allow_pickle=True).item()
        if (np.array(data["array_of_irradiance_field_clearsky"]).shape)[0] != 2:
            print(file_dir + file_superscript + '1586201895278.0117_3021608184182.0234_' + file_subscript + '.npy')

        array_of_irradiance_field_clearsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))
        array_of_irradiance_field_allsky = np.zeros_like(np.array(data["array_of_irradiance_field_clearsky"]))


        for i in np.arange(0, 29):
            frequencies_lw = convert.kaycm2freq(wavenumbers_lw)
            frequencies_lw = frequencies_lw[i*1597:i*1597+1597]
            #dnu = np.max(frequencies_sw) - np.min(frequencies_sw)
            data = np.load(file_dir + file_superscript + str(np.min(frequencies_lw)) + '_' + str(np.max(frequencies_lw)) + '_'  + file_subscript + '.npy', allow_pickle=True).item()
            if (np.array(data["array_of_irradiance_field_clearsky"]).shape)[0] != 2:
                print(file_dir + file_superscript + str(np.min(frequencies_lw)) + '_' + str(np.max(frequencies_lw)) + '_'  + file_subscript + '.npy')
    
    else:
        print("regime must be either sw or lw")

    return None


def concat_arts_output_single_atm(file_dir, file_superscript, file_subscript, regime, allsky = True):

    if regime == 'sw': 
        d1, d2, d3, d4, d5 = 0.002, 0.001, 0.002, 0.02, 1
        m = 100
        ## set wavenumbers at fine spectral discretization
        wavenumbers_sw = np.concatenate([np.arange(250, 2200, m*d1), 
                        np.arange(2200, 2400, m*d2),
                        np.arange(2400, 5150, m*d3),
                        np.arange(5150, 16000, m*d4),
                        #np.arange(16000, 35000, d5)])
                        np.arange(16000, 48979, d5),
                        np.arange(48979, 49950, d5)])
        data = np.load(file_dir + file_superscript + '989405049137400.0_1049243623754200.0_' + file_subscript + '.npy', allow_pickle=True).item()
        flux_clearsky_up = np.zeros_like(np.array(data["flux_clearsky_up"]))
        flux_clearsky_down = np.zeros_like(np.array(data["flux_clearsky_down"]))
        flux_allsky_up = np.zeros_like(np.array(data["flux_clearsky_up"]))
        flux_allsky_down = np.zeros_like(np.array(data["flux_clearsky_down"]))

        for i in np.arange(0, 33):
            frequencies_sw = convert.kaycm2freq(wavenumbers_sw)
            if i == 32:
                frequencies_sw = frequencies_sw[i*1997:]

            else:
                frequencies_sw = frequencies_sw[i*1997:i*1997+1997]
            #dnu = np.max(frequencies_sw) - np.min(frequencies_sw)
            #print(file_dir + file_superscript + str(np.min(frequencies_sw)) + '_' + str(np.max(frequencies_sw)) + '_' + file_subscript + '.npy')
            data = np.load(file_dir + file_superscript + str(np.min(frequencies_sw)) + '_' + str(np.max(frequencies_sw)) + '_' + file_subscript + '.npy', allow_pickle=True).item()
            if allsky == True:
                flux_allsky_up += np.array(data["flux_allsky_up"])
                flux_allsky_down += np.array(data["flux_allsky_down"])
            flux_clearsky_up += np.array(data["flux_clearsky_up"])
            flux_clearsky_down += np.array(data["flux_clearsky_down"])

        results = xr.Dataset(
            data_vars = dict(
                flux_cs = (["pressure", "direction"], np.stack([flux_clearsky_down, flux_clearsky_up], axis=-1)),
                flux_as = (["pressure", "direction"], np.stack([flux_allsky_down, flux_allsky_up], axis=-1)),
            ),
            coords = dict(
                pressure = np.array(data["pressure"]),
                direction = np.array(["down", "up"])
            ),
        )

    elif regime == 'lw':
        d1, d2, d3 = 0.0002, 0.001, 0.005
        m = 150
        ## set wavenumbers at fine spectral discretization
        wavenumbers_lw = np.concatenate([np.arange(5, 1300, m*d1), 
                        np.arange(1300, 1700, m*d2),
                        np.arange(1700, 3257, m*d3)])

        data = np.load(file_dir + file_superscript + '1586201895278.0117_3021608184182.0234_' + file_subscript + '.npy', allow_pickle=True).item()
        flux_clearsky_up = np.zeros_like(np.array(data["flux_clearsky_up"]))
        flux_clearsky_down = np.zeros_like(np.array(data["flux_clearsky_down"]))
        flux_allsky_up = np.zeros_like(np.array(data["flux_clearsky_up"]))
        flux_allsky_down = np.zeros_like(np.array(data["flux_clearsky_down"]))


        for i in np.arange(0, 29):
            frequencies_lw = convert.kaycm2freq(wavenumbers_lw)
            frequencies_lw = frequencies_lw[i*1597:i*1597+1597]
            #dnu = np.max(frequencies_sw) - np.min(frequencies_sw)
            data = np.load(file_dir + file_superscript + str(np.min(frequencies_lw)) + '_' + str(np.max(frequencies_lw)) + '_'  + file_subscript + '.npy', allow_pickle=True).item()
            if allsky == True:
                if 'flux_allsky_up' in data.keys():
                    flux_allsky_up += np.array(data["flux_allsky_up"])
                    flux_allsky_down += np.array(data["flux_allsky_down"])
                else:
                    flux_allsky_up += np.array(data["flux_clearsky_up"])
                    flux_allsky_down += np.array(data["flux_clearsky_down"])

            flux_clearsky_up += np.array(data["flux_clearsky_up"])
            flux_clearsky_down += np.array(data["flux_clearsky_down"])

        results = xr.Dataset(
            data_vars = dict(
                flux_cs = (["pressure", "direction"], np.stack([flux_clearsky_down, flux_clearsky_up], axis=-1)),
                flux_as = (["pressure", "direction"], np.stack([flux_allsky_down, flux_allsky_up], axis=-1)),
            ),
            coords = dict(
                pressure = np.array(data["pressure"]),
                direction = np.array(["down", "up"])
            ),

        )
    
    else:
        print("regime must be either sw or lw")

    return results