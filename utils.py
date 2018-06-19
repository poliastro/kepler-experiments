from poliastro.twobody.angles import E_to_nu, F_to_nu, nu_to_E, nu_to_F
import numpy as np
from scipy.optimize import newton
from astropy import units as u

variation_changes_sign = []
true_anomalies = []
variation = 1e-12


def find_end_point(variation_changes_sign, true_anomalies):
    for i in range(len(true_anomalies) - 1):
        if np.abs(true_anomalies[i + 1] - true_anomalies[i]) < variation and variation_changes_sign[i + 1]:
            return i + 1
    return -1


def _kepler_equation(E, M, ecc):
    global variation_changes_sign, true_anomalies
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        nu = E_to_nu(E, ecc)
        E_plus = nu_to_E(nu + variation, ecc)
        E_minus = nu_to_E(nu - variation, ecc)
        M_actual = E - ecc * np.sin(E)
        M_plus = E_plus - ecc * np.sin(E_plus)
        M_minus = (E_minus) - ecc * np.sin(E_minus)

        true_anomalies.append(nu)
        variation_changes_sign.append(M_actual * M_plus * M_minus < 0)
        return E - ecc * np.sin(E) - M


def _kepler_equation_prime(E, M, ecc):
    return 1 - ecc * np.cos(E)


def _kepler_equation_hyper(F, M, ecc):
    global variation_changes_sign, true_anomalies
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        nu = F_to_nu(F, ecc)
        F_plus = nu_to_F(nu + variation, ecc)
        F_minus = nu_to_F(nu - variation, ecc)
        M_actual = -F + ecc * np.sinh(F)
        M_plus = -F_plus + ecc * np.sinh(F_plus)
        M_minus = -F_minus + ecc * np.sinh(F_minus)

        true_anomalies.append(nu)
        variation_changes_sign.append(M_actual * M_plus * M_minus < 0)
        return -F + ecc * np.sinh(F) - M


def _kepler_equation_prime_hyper(F, M, ecc):
    return ecc * np.cosh(F) - 1

def test_inversion(ecc, delta_t):
    global true_anomalies, variation_changes_sign
    true_anomalies, variation_changes_sign = [], []
    if ecc < 1:
        # elliptic case
        M = np.sqrt((1.0 - ecc) ** 3) * delta_t
        # the way it is done currently in Poliastro
        E = newton(_kepler_equation, M, _kepler_equation_prime, args=(M, ecc), maxiter=1000)
    else:
        M = np.sqrt((ecc - 1) ** 3) * delta_t
        # the way it is done currently in Poliastro
        F = newton(_kepler_equation_hyper, np.arcsinh(M / ecc), _kepler_equation_prime_hyper,
                            args=(M, ecc), maxiter=1000)

    return find_end_point(variation_changes_sign, true_anomalies)


def run_all_tests(eccs, delta_ts):
    eccs, delta_ts = np.meshgrid(eccs, delta_ts)
    num_iters = []
    for ecc, delta_t in zip(eccs.flatten(), delta_ts.flatten()):
        num_iters.append(test_inversion(ecc, delta_t))
