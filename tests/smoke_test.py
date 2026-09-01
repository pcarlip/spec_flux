from spectral_flux.advection import advection_xr
from spectral_flux.coarse_calc import pi_cg_lst_xr
from spectral_flux.fourier_calc import (
    fourier_int_xr_lst,
    fourier_prep_xr,
    van_atta_int,
    van_atta_prep,
)
from spectral_flux.sf_au import sf_au_dir_xr, sf_au_xr
from spectral_flux.sf_ln import sf_ln_dir_xr, sf_ln_xr

# basic test that I can import the modules at all


def smoke_test():
    return
