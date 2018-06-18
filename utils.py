from poliastro.angles import E_to_nu, F_to_nu, nu_to_E, nu_to_F
import numpy as np
from scipy.optimize import newton
from astropy import units as u

variation_changes_sign = []
true_anomalies = []
variation = 1e-12

def _kepler_equation(E, M, ecc):
    global variation_changes_sign
    global true_anomalies
    
    nu = E_to_nu(nu)
    E_plus = nu_to_E(nu + variation)
    E_minus = nu_to_E(nu - variation)
    M_actual = E - ecc * np.sin(E)
    M_plus = E_plus - ecc * np.sin(E_plus)
    M_minus = (E_minus) - ecc * np.sin(E_minus)

    true_anomalies.append(nu)
    variation_changes_sign.append(M_actual * M_plus * M_minus < 0)
    return E - ecc * np.sin(E) - M


def _kepler_equation_prime(E, M, ecc):
    return 1 - ecc * np.cos(E)


def _kepler_equation_hyper(F, M, ecc):
    global variation_changes_sign
    global true_anomalies
    
    nu = F_to_nu(nu)
    F_plus = nu_to_F(nu + variation)
    F_minus = nu_to_F(nu - variation)
    M_actual = -F + ecc * np.sinh(F)
    M_plus = -F_plus + ecc * np.sinh(F_plus)
    M_minus = -F_minus + ecc * np.sinh(F_minus)

    true_anomalies.append(nu)
    variation_changes_sign.append(M_actual * M_plus * M_minus < 0)
    return -F + ecc * np.sinh(F) - M


def _kepler_equation_prime_hyper(F, M, ecc):
    return ecc * np.cosh(F) - 1

# def test_inversion(ecc, )