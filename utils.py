from poliastro.twobody.angles import E_to_nu, F_to_nu, nu_to_E, nu_to_F
import numpy as np
from scipy.optimize import newton
from astropy import units as u
num_calls = 0
variation = 1e-12


def _kepler_equation(E, M, ecc):
    global num_calls
    num_calls += 1
    return E - ecc * np.sin(E) - M


def _kepler_equation_prime(E, M, ecc):
    return 1 - ecc * np.cos(E)


def _kepler_equation_hyper(F, M, ecc):
    global num_calls
    num_calls += 1
    return -F + ecc * np.sinh(F) - M


def _kepler_equation_prime_hyper(F, M, ecc):
    return ecc * np.cosh(F) - 1

def test_inversion(ecc, delta_t):
    global num_calls
    num_calls = 0
    if ecc < 1:
        # elliptic case
        M = np.sqrt((1.0 - ecc) ** 3) * delta_t
        # the way it is done currently in Poliastro
        E = newton(_kepler_equation, M, _kepler_equation_prime, args=(M, ecc), maxiter=1000, tol=1e-12)
    else:
        M = np.sqrt((ecc - 1) ** 3) * delta_t
        # the way it is done currently in Poliastro
        F = newton(_kepler_equation_hyper, np.arcsinh(M / ecc), _kepler_equation_prime_hyper,
                            args=(M, ecc), maxiter=1000, tol=1e-12)

    return num_calls

def run_all_tests(eccs, delta_ts):
    eccs, delta_ts = np.meshgrid(eccs, delta_ts)
    num_iters = []

    for ecc, delta_t in zip(eccs.flatten(), delta_ts.flatten()):
        num_iters.append(test_inversion(ecc, delta_t))
    return eccs.flatten(), delta_ts.flatten(), np.array(num_iters)