from poliastro.twobody.angles import E_to_nu, F_to_nu, nu_to_E, nu_to_F
from joblib import Parallel, delayed
import numpy as np
from astropy import units as u
from tqdm import tqdm
num_calls = 0
variation = 1e-12


def newton(func, x0, ecc, M, fprime=None, maxiter=50):
    EF = 1.0 * x0
    nu_prev = 1e+10
    converged = False
    # Newton-Rapheson method
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        for iter in range(maxiter):
            if ecc < 1.0:
                nu = E_to_nu(EF, ecc)
                E_plus = nu_to_E(nu + variation, ecc)
                E_minus = nu_to_E(nu - variation, ecc)
                M_actual = _kepler_equation(EF, M, ecc, count=False)
                M_plus = _kepler_equation(E_plus, M, ecc, count=False)
                M_minus = _kepler_equation(E_minus, M, ecc, count=False)
            else:
                nu = F_to_nu(EF, ecc)
                F_plus = nu_to_F(nu + variation, ecc)
                F_minus = nu_to_F(nu - variation, ecc)
                M_actual = _kepler_equation_hyper(EF, M, ecc, count=False)
                M_plus = _kepler_equation_hyper(F_plus, M, ecc, count=False)
                M_minus = _kepler_equation_hyper(F_minus, M, ecc, count=False)
            converged = (np.abs(nu_prev - nu) < variation) and (M_actual * M_plus <= 0 or M_minus * M_actual <= 0)
            # print(ecc, nu - nu_prev, M_actual, M_plus, M_minus, M)
            nu_prev = nu
            EF_new = EF - func(EF, M, ecc) / fprime(EF, M, ecc)
            if converged:
                return EF_new
            EF = EF_new
    print(ecc, "fail")
    return -1


def _kepler_equation(E, M, ecc, count=True):
    global num_calls
    if count:
        num_calls += 1
    return E - ecc * np.sin(E) - M


def _kepler_equation_prime(E, M, ecc):
    return 1 - ecc * np.cos(E)


def _kepler_equation_hyper(F, M, ecc, count=True):
    global num_calls
    if count:
        num_calls += 1
    return -F + ecc * np.sinh(F) - M


def _kepler_equation_prime_hyper(F, M, ecc):
    return ecc * np.cosh(F) - 1

def test_inversion(ecc, delta_t, initial_guess):
    global num_calls
    num_calls = 0
    if ecc < 1:
        # elliptic case
        M = np.sqrt((1.0 - ecc) ** 3) * delta_t
        # the way it is done currently in Poliastro
        # np.pi is the starting point chosen in the paper
        if initial_guess == 'paper':
            guess = np.pi
        else:
            guess = M
        E = newton(_kepler_equation, guess, ecc, M, _kepler_equation_prime, maxiter=100)
    else:
        M = np.sqrt((ecc - 1) ** 3) * delta_t
        # the way it is done currently in Poliastro
        
        if initial_guess == 'paper':
            C = np.exp(1.0) * (ecc + 2 * M) / (ecc * np.exp(1) - 2)
            guess = np.min([np.log(C), np.arcsinh(M / (ecc - 1.0))])
        else:
            guess = np.arcsinh(M / ecc)
        F = newton(_kepler_equation_hyper, guess, ecc, M, _kepler_equation_prime_hyper, maxiter=100)

    return num_calls

def run_all_tests(eccs, delta_ts, initial_guess):
    eccs, delta_ts = np.meshgrid(eccs, delta_ts)
    num_iters = []

    eccs = eccs.flatten()
    delta_ts = delta_ts.flatten()
    num_iters = Parallel(n_jobs=-1)(delayed(test_inversion)(ecc, delta_t, initial_guess) for ecc, delta_t in zip(eccs, delta_ts))
    return eccs.flatten(), delta_ts.flatten(), np.array(num_iters)