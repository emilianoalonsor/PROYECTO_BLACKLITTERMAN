

import numpy as np
import pandas as pd


# ---------------------------
# Las siguientes funciones se encargan de revisar que los datos sean coherentes para trabajarlos.
# ---------------------------

def _to_numpy_1d(x, name: str):
    """Convierte Series/list/ndarray a np.ndarray 1D."""
    if isinstance(x, pd.Series):
        x = x.values
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} debe ser un arreglo 1D.")
    return x


def _to_numpy_2d(x, name: str):
    """Convierte DataFrame/list/ndarray a np.ndarray 2D."""
    if isinstance(x, pd.DataFrame):
        x = x.values
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} debe ser una matriz 2D.")
    return x


def _normalize_weights(w: np.ndarray):
    """Normaliza pesos para que sumen 1."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if s == 0:
        raise ValueError("La suma de pesos es 0; no se puede normalizar.")
    return w / s


def _check_dimensions(Sigma, w_mkt, P=None, Q=None, Omega=None):
    """Revisa dimensiones consistentes."""
    Sigma = _to_numpy_2d(Sigma, "Sigma")
    n = Sigma.shape[0]
    if Sigma.shape[1] != n:
        raise ValueError("Sigma debe ser cuadrada (N×N).")

    w_mkt = _to_numpy_1d(w_mkt, "w_mkt")
    if w_mkt.shape[0] != n:
        raise ValueError("w_mkt debe tener tamaño N, consistente con Sigma.")

    if P is not None:
        P = _to_numpy_2d(P, "P")
        if P.shape[1] != n:
            raise ValueError("P debe tener N columnas (K×N).")

        if Q is None:
            raise ValueError("Si pasas P, también debes pasar Q.")
        Q = _to_numpy_1d(Q, "Q")
        if Q.shape[0] != P.shape[0]:
            raise ValueError("Q debe tener tamaño K, consistente con P (K×N).")

        if Omega is not None:
            Omega = _to_numpy_2d(Omega, "Omega")
            if Omega.shape != (P.shape[0], P.shape[0]):
                raise ValueError("Omega debe ser K×K, consistente con P y Q.")

    return Sigma, w_mkt


# ---------------------------
# Función que construye las views absolutas
# ---------------------------

def view_absoluta(tickers: list[str], ticker: str, q: float):
    """
    Construye una view absoluta:
    "ticker tendrá retorno (exceso) q".
    P_k: 1 en ese ticker, 0 en el resto.
    """
    if ticker not in tickers:
        raise ValueError(f"{ticker} no está en el universo.")
    n = len(tickers)
    row = np.zeros(n, dtype=float)
    row[tickers.index(ticker)] = 1.0
    return row, float(q)

# ---------------------------
# Función que construye las views relativas
# ---------------------------
def view_relativa(tickers: list[str], ticker_long: str, ticker_short: str, q: float):
    """
    Construye una view relativa:
    "ticker_long superará a ticker_short por q".
    P_k: +1 en long, -1 en short, 0 en el resto.
    """
    if ticker_long not in tickers:
        raise ValueError(f"{ticker_long} no está en el universo.")
    if ticker_short not in tickers:
        raise ValueError(f"{ticker_short} no está en el universo.")
    if ticker_long == ticker_short:
        raise ValueError("En una view relativa, long y short deben ser diferentes.")

    n = len(tickers)
    row = np.zeros(n, dtype=float)
    row[tickers.index(ticker_long)] = 1.0
    row[tickers.index(ticker_short)] = -1.0
    return row, float(q)

# ---------------------------
# Función que apila los views
# ---------------------------
def apilar_views(rows_P: list[np.ndarray], vals_Q: list[float]):
    
    if len(rows_P) == 0:
        return np.zeros((0, 0)), np.zeros((0,))
    P = np.vstack([np.asarray(r, dtype=float) for r in rows_P])
    Q = np.asarray(vals_Q, dtype=float).reshape(-1)
    if P.shape[0] != Q.shape[0]:
        raise ValueError("Número de filas de P y tamaño de Q no coinciden.")
    return P, Q


# ---------------------------
# Función que calcula tau
# ---------------------------

def tau_por_T(T: int):
  
    if T is None or T <= 0:
        raise ValueError("T debe ser un entero positivo.")
    return 1.0 / float(T)

# ---------------------------
# Función que estima delta
# ---------------------------
def estimar_delta(Sigma, w_mkt, mu_anual_mkt: float, rf_anual: float):

    Sigma = _to_numpy_2d(Sigma, "Sigma")
    w_mkt = _normalize_weights(_to_numpy_1d(w_mkt, "w_mkt"))
    var_mkt = float(w_mkt @ Sigma @ w_mkt)

    if var_mkt <= 0:
        raise ValueError("Varianza del benchmark no positiva; revisa Sigma.")

    excess_mkt = float(mu_anual_mkt - rf_anual)
    return excess_mkt / var_mkt

# ---------------------------
# Función que calcula el parámetro pi
# ---------------------------
def prior_pi(Sigma, w_mkt, delta: float):

    Sigma = _to_numpy_2d(Sigma, "Sigma")
    w_mkt = _normalize_weights(_to_numpy_1d(w_mkt, "w_mkt"))
    return float(delta) * (Sigma @ w_mkt)

# ---------------------------
# Función que construye la matriz omega
# ---------------------------
def omega_he_litterman(P, Sigma, tau: float, conf=None, eps: float = 1e-12):
   
    P = _to_numpy_2d(P, "P")
    Sigma = _to_numpy_2d(Sigma, "Sigma")
    tau = float(tau)

    if P.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)

    tauSigma = tau * Sigma
    base = P @ tauSigma @ P.T
    base_diag = np.clip(np.diag(base), eps, None)  

    if conf is None:
        omega_diag = base_diag
    else:
        conf = _to_numpy_1d(conf, "conf")
        if conf.shape[0] != P.shape[0]:
            raise ValueError("conf debe tener tamaño K (una confianza por view).")
        
        conf = np.clip(conf, 1e-6, 1.0)
        omega_diag = base_diag / conf

    return np.diag(omega_diag)


# ---------------------------
# Función que calcula el BL posterior
# ---------------------------

def posterior_bl(Sigma, pi, P, Q, Omega, tau: float, ridge: float = 1e-10):

    Sigma = _to_numpy_2d(Sigma, "Sigma")
    pi = _to_numpy_1d(pi, "pi")
    P = _to_numpy_2d(P, "P")
    Q = _to_numpy_1d(Q, "Q")
    Omega = _to_numpy_2d(Omega, "Omega")

    n = Sigma.shape[0]
    k = P.shape[0]
    if pi.shape[0] != n:
        raise ValueError("pi debe tener tamaño N.")
    if P.shape[1] != n:
        raise ValueError("P debe ser K×N.")
    if Q.shape[0] != k:
        raise ValueError("Q debe tener tamaño K.")
    if Omega.shape != (k, k):
        raise ValueError("Omega debe ser K×K.")

    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau debe ser > 0.")

    if k == 0:
        mu_post = pi.copy()
        Sigma_post = tau * Sigma
        return mu_post, Sigma_post


    tauSigma = tau * Sigma
    tauSigma = tauSigma + ridge * np.eye(n)

    inv_tauSigma = np.linalg.solve(tauSigma, np.eye(n))


    Omega_reg = Omega + ridge * np.eye(k)
    inv_Omega = np.linalg.solve(Omega_reg, np.eye(k))

    A = inv_tauSigma + P.T @ inv_Omega @ P
    A = A + ridge * np.eye(n)

    b = inv_tauSigma @ pi + P.T @ inv_Omega @ Q

    mu_post = np.linalg.solve(A, b)
    Sigma_post = np.linalg.solve(A, np.eye(n))

    return mu_post, Sigma_post


#Construimos la función principal para mandarla a llamar en app.py

def black_litterman(
    Sigma,
    w_mkt,
    rf_anual: float,
    mu_anual=None,
    P=None,
    Q=None,
    conf=None,
    tau: float | None = None,
    delta: float | None = None,
    ridge: float = 1e-10,
):

    Sigma, w_mkt = _check_dimensions(Sigma, w_mkt)

    n = Sigma.shape[0]
    w_mkt = _normalize_weights(_to_numpy_1d(w_mkt, "w_mkt"))

    rf_anual = float(rf_anual)

    
    if tau is None:
        tau = 0.025
    tau = float(tau)

    if delta is None:
        if mu_anual is None:
            raise ValueError("Si delta=None, debes pasar mu_anual para estimarlo.")
        mu_anual = _to_numpy_1d(mu_anual, "mu_anual")
        if mu_anual.shape[0] != n:
            raise ValueError("mu_anual debe tener tamaño N.")
        mu_mkt = float(w_mkt @ mu_anual)
        delta = estimar_delta(Sigma, w_mkt, mu_mkt, rf_anual)
    delta = float(delta)


    pi = prior_pi(Sigma, w_mkt, delta)

  
    if P is None or Q is None:
        mu_excess = pi
        Sigma_post = tau * _to_numpy_2d(Sigma, "Sigma")
        mu_total = mu_excess + rf_anual
        Omega = np.zeros((0, 0), dtype=float)
        return mu_excess, mu_total, Sigma_post, pi, Omega, delta, tau


    P = _to_numpy_2d(P, "P")
    Q = _to_numpy_1d(Q, "Q")
    if P.shape[1] != n:
        raise ValueError("P debe tener N columnas.")
    if Q.shape[0] != P.shape[0]:
        raise ValueError("Q debe tener tamaño K, consistente con P.")

  
    Omega = omega_he_litterman(P, Sigma, tau=tau, conf=conf)


    mu_excess, Sigma_post = posterior_bl(
        Sigma=Sigma,
        pi=pi,
        P=P,
        Q=Q,
        Omega=Omega,
        tau=tau,
        ridge=ridge,
    )

    mu_total = mu_excess + rf_anual
    return mu_excess, mu_total, Sigma_post, pi, Omega, delta, tau
