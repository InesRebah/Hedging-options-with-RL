""" Utility Functions """

'''
utils.py sert à générer toutes les données financières utilisées par l’environnement RL :
1. simuler des trajectoires de prix S_t
2. calculer le prix de l’option V_t
3. calculer le delta Black-Scholes
4. simuler le modèle SABR / volatilité stochastique
5. calculer le delta de Bartlett

Donc ce fichier correspond aux sections 4 et 5 du papier :
section 4 : GBM + Black-Scholes ;
section 5 : SABR / volatilité stochastique + implied vol + Bartlett delta.
fct: brownian_sim, bs_call, get_sim_path, sabr_sim, sabr_implied_vol, bartlett, get_sim_path_sabr
'''
import random
import numpy as np
from scipy.stats import norm

random.seed(1)

def brownian_sim(num_path, num_period, mu, std, init_p, dt):
    #num paths nb de paths simules, num period : nbr de dates dans chaque trajectoire 
    z = np.random.normal(size=(num_path, num_period))

    a_price = np.zeros((num_path, num_period))
    a_price[:, 0] = init_p

    for t in range(num_period - 1):
        a_price[:, t + 1] = a_price[:, t] * np.exp(
            (mu - (std ** 2) / 2) * dt + std * np.sqrt(dt) * z[:, t]
        )
    return a_price


# BSM Call Option Pricing Formula & BS Delta formula
# T here is time to maturity
# def bs_call(iv, T, S, K, r, q):
#     d1 = (np.log(S / K) + (r - q + iv * iv / 2) * T) / (iv * np.sqrt(T))
#     d2 = d1 - iv * np.sqrt(T)
#     bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     bs_delta = np.exp(-q * T) * norm.cdf(d1)
#     return bs_price, bs_delta

def bs_call(iv, T, S, K, r, q):
    T_safe = np.maximum(T, 1e-8)

    d1 = (np.log(S / K) + (r - q + iv * iv / 2) * T_safe) / (iv * np.sqrt(T_safe))
    d2 = d1 - iv * np.sqrt(T_safe)

    bs_price = S * np.exp(-q * T_safe) * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    bs_delta = np.exp(-q * T_safe) * norm.cdf(d1)

    payoff = np.maximum(S - K, 0.0)

    bs_price = np.where(T <= 0, payoff, bs_price)
    bs_delta = np.where(T <= 0, (S > K).astype(float), bs_delta)

    return bs_price, bs_delta

def get_sim_path(M, freq, np_seed, num_sim):
    """ Return simulated data: a tuple of three arrays
        M: initial time to maturity
        freq: trading freq in unit of day, e.g. freq=2: every 2 day; freq=0.5 twice a day;
        np_seed: numpy random seed
        num_sim: number of simulation path

        1) asset price paths (num_path x num_period)
        2) option price paths (num_path x num_period)
        3) delta (num_path x num_period)
    """
    # set the np random seed
    np.random.seed(np_seed)

    # Trading Freq per day; passed from function parameter
    # freq = 2

    # Annual Trading Day
    T = 250

    # Simulation Time Step
    dt = 0.004 * freq

    # Option Day to Maturity; passed from function parameter
    # M = 60

    # Number of period
    num_period = int(M / freq)

    # Number of simulations; passed from function parameter
    # num_sim = 1000000

    # Annual Return
    mu = 0.05

    # Annual Volatility
    vol = 0.2

    # Initial Asset Value
    S = 100

    # Option Strike Price
    K = 100

    # Annual Risk Free Rate
    r = 0

    # Annual Dividend
    q = 0

    # asset price 2-d array
    print("1. generate asset price paths")
    a_price = brownian_sim(num_sim, num_period + 1, mu, vol, S, dt)

    # time to maturity "rank 1" array: e.g. [M, M-1, ..., 0]
    ttm = np.arange(M, -freq, -freq)

    # BS price 2-d array and bs delta 2-d array
    print("2. generate BS price and delta")
    bs_price, bs_delta = bs_call(vol, ttm / T, a_price, K, r, q)

    print("simulation done!")

    return a_price, bs_price, bs_delta


def sabr_sim(num_path, num_period, mu, std, init_p, dt, rho, beta, volvol):
    qs = np.random.normal(size=(num_path, num_period))
    qi = np.random.normal(size=(num_path, num_period))
    qv = rho * qs + np.sqrt(1 - rho * rho) * qi

    vol = np.zeros((num_path, num_period))
    vol[:, 0] = std

    a_price = np.zeros((num_path, num_period))
    a_price[:, 0] = init_p

    for t in range(num_period - 1):
        gvol = vol[:, t] * (a_price[:, t] ** (beta - 1))
        a_price[:, t + 1] = a_price[:, t] * np.exp(
            (mu - (gvol ** 2) / 2) * dt + gvol * np.sqrt(dt) * qs[:, t]
        )
        vol[:, t + 1] = vol[:, t] * np.exp(
            -volvol * volvol * 0.5 * dt + volvol * qv[:, t] * np.sqrt(dt)
        )

    return a_price, vol


def sabr_implied_vol(vol, T, S, K, r, q, beta, volvol, rho):

    F = S * np.exp((r - q) * T)
    x = (F * K) ** ((1 - beta) / 2)
    y = (1 - beta) * np.log(F / K)
    A = vol / (x * (1 + y * y / 24 + y * y * y * y / 1920))
    B = 1 + T * (
        ((1 - beta) ** 2) * (vol * vol) / (24 * x * x)
        + rho * beta * volvol * vol / (4 * x)
        + volvol * volvol * (2 - 3 * rho * rho) / 24
    )
    Phi = (volvol * x / vol) * np.log(F / K)
    Chi = np.log((np.sqrt(1 - 2 * rho * Phi + Phi * Phi) + Phi - rho) / (1 - rho))

    SABRIV = np.where(F == K, vol * B / (F ** (1 - beta)), A * B * Phi / Chi)

    return SABRIV


def bartlett(sigma, T, S, K, r, q, ds, beta, volvol, rho):

    dsigma = ds * volvol * rho / (S ** beta)

    vol1 = sabr_implied_vol(sigma, T, S, K, r, q, beta, volvol, rho)
    vol2 = sabr_implied_vol(sigma + dsigma, T, S + ds, K, r, q, beta, volvol, rho)

    bs_price1, _ = bs_call(vol1, T, S, K, r, q)
    bs_price2, _ = bs_call(vol2, T, S+ds, K, r, q)

    b_delta = (bs_price2 - bs_price1) / ds

    return b_delta



def get_sim_path_sabr(
                        M,
                        freq,       # trading freq per day
                        np_seed,
                        num_sim,
                        mu=0.05,
                        vol=0.2,
                        beta=1.0,
                        rho=-0.4,
                        volvol=0.6,
                        S=100,
                        K=100,
                        r=0,
                        q=0,
                        ds=0.001,
                    ):
    """ Return simulated data: a tuple of four arrays
        M: initial time to maturity
        freq: trading freq in unit of day, e.g. freq=2: every 2 day; freq=0.5 twice a day;
        np_seed: numpy random seed
        num_sim: number of simulation path

        1) asset price paths (num_path x num_period)
        2) option price paths (num_path x num_period)
        3) bs delta (num_path x num_period)
        4) bartlett delta (num_path x num_period)
    """
    # set the np random seed
    np.random.seed(np_seed)


    # Annual Trading Day
    T = 250

    # Simulation Time Step
    dt = 0.004 * freq

    # Option Day to Maturity; passed from function parameter
    # M = 60

    # Number of period
    num_period = int(M / freq)

    # Number of simulations; passed from function parameter
    # num_sim = 1000000

    # asset price 2-d array; sabr_vol
    print("1. generate asset price paths (sabr)")
    a_price, sabr_vol = sabr_sim(
        num_sim, num_period + 1, mu, vol, S, dt, rho, beta, volvol
    )

    # time to maturity "rank 1" array: e.g. [M, M-1, ..., 0]
    ttm = np.arange(M, -freq, -freq)

    # BS price 2-d array and bs delta 2-d array
    print("2. generate BS price, BS delta, and Bartlett delta")

    # sabr implied vol
    implied_vol = sabr_implied_vol(
        sabr_vol, ttm / T, a_price, K, r, q, beta, volvol, rho
    )

    bs_price, bs_delta = bs_call(implied_vol, ttm / T, a_price, K, r, q)

    bartlett_delta = bartlett(sabr_vol, ttm / T, a_price, K, r, q, ds, beta, volvol, rho)

    print("simulation done!")

    return a_price, bs_price, bs_delta, bartlett_delta
#####FX
def garman_kohlhagen_call(vol, T, S, K, rd, rf):
    T_safe = np.maximum(T, 1e-8)

    d1 = (np.log(S / K) + (rd - rf + 0.5 * vol**2) * T_safe) / (vol * np.sqrt(T_safe))
    d2 = d1 - vol * np.sqrt(T_safe)

    price = S * np.exp(-rf * T_safe) * norm.cdf(d1) - K * np.exp(-rd * T_safe) * norm.cdf(d2)
    delta = np.exp(-rf * T_safe) * norm.cdf(d1)

    payoff = np.maximum(S - K, 0.0)

    price = np.where(T <= 0, payoff, price)
    delta = np.where(T <= 0, (S > K).astype(float), delta)

    return price, delta

def get_sim_path_fx(M,freq,np_seed,num_sim,mu=0.02,vol=0.10,S=100,K=100,rd=0.04,rf=0.02):
    np.random.seed(np_seed)

    T = 250
    dt = freq / T
    num_period = int(M / freq)

    print("1. generate FX paths")
    fx_path = brownian_sim(num_sim, num_period + 1, mu, vol, S, dt)

    ttm = np.arange(M, -freq, -freq)

    print("2. generate Garman-Kohlhagen price and delta")
    option_price, delta = garman_kohlhagen_call(
        vol,
        ttm / T,
        fx_path,
        K,
        rd,
        rf
    )

    print("FX simulation done!")

    return fx_path, option_price, delta

def log_ou_sim(num_path, num_period, kappa, theta, sigma, init_s, dt):
    z = np.random.normal(size=(num_path, num_period))

    x = np.zeros((num_path, num_period))
    x[:, 0] = np.log(init_s)

    for t in range(num_period - 1):
        x[:, t + 1] = (
            x[:, t]
            + kappa * (theta - x[:, t]) * dt
            + sigma * np.sqrt(dt) * z[:, t]
        )

    return np.exp(x)


def get_sim_path_commodity(
    M,
    freq,
    np_seed,
    num_sim,
    kappa=2.0,
    theta=np.log(100),
    sigma=0.30,
    S=100,
    K=100,
    r=0.04,
    q=0.03,
    sigma_bs=0.30,
):
    np.random.seed(np_seed)

    T = 250
    dt = freq / T
    num_period = int(M / freq)

    print("1. generate commodity log-OU paths")
    commodity_path = log_ou_sim(
        num_path=num_sim,
        num_period=num_period + 1,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        init_s=S,
        dt=dt,
    )

    ttm = np.arange(M, -freq, -freq)

    print("2. generate BS proxy price and delta with convenience yield")
    option_price, delta = bs_call(
        sigma_bs,
        ttm / T,
        commodity_path,
        K,
        r,
        q,
    )

    print("Commodity simulation done!")

    return commodity_path, option_price, delta