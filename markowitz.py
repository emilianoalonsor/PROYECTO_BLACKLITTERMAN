import numpy as np 
from scipy.optimize import minimize

#---------------
'''
Este módulo se encarga de:
-Encontrar portafolio de mínima varianza
-Encontrar portafolio de máxima sharpe
-Encontrar portafolio con rendimiento objetivo
'''
#---------------

#---------------Función auxiliar
def perf_portafolio(pesos, media_anual,cov_anual, tasa_libre_anual):
    pesos = np.array(pesos, dtype=float)
    if pesos.ndim != 1:
        raise ValueError('La dimensión del vector de pesos debe ser 1.')
    suma = pesos.sum()
    if suma == 0:
        raise ValueError('La suma de pesos es 0.')
    w = pesos/suma
    rend = np.dot(w,media_anual)
    vol = np.sqrt(w @ cov_anual @ w)
    if vol > 0:
        sharpe = (rend - tasa_libre_anual)/vol
    else:
        sharpe = np.nan
    return rend,vol,sharpe
#---------------

#---------------Función que encuentra el portafolio de mínima varianza
def port_min_varianza(media_anual,cov_anual):
    n = len(media_anual)
    x0 = np.repeat(1/n,n) #Vector equiponderado
    bounds = tuple((0.0,1.0) for _ in range(n))
    cons = {'type':'eq','fun':lambda w: np.sum(w)-1}
    def obj(w):
        w = np.array(w,dtype=float)
        return float(w @ cov_anual  @ w)
    res = minimize(obj, x0, method='SLSQP', bounds=bounds,constraints=cons) 
    if not res.success:
        raise ValueError(f'Optimización de minima varianza no convergió:{res.message}')
    w_opt = np.array(res.x, dtype=float)
    w_opt = w_opt/w_opt.sum()
    return w_opt
#---------------
    
#---------------Función que encuentra el portafolio de máximo Sharpe
def port_sharpe(media_anual,cov_anual,tasa_libre_anual):
    n = len(media_anual)
    x0 = np.repeat(1/n,n)
    bounds = tuple((0.0,1.0) for _ in range(n))
    cons = {'type':'eq','fun': lambda w: np.sum(w)-1}
    def obj(w):
        rend, vol, sharpe = perf_portafolio(w,media_anual,cov_anual,tasa_libre_anual)
        if np.isnan(sharpe):
            return 1e6
        return -sharpe
    res = minimize(obj,x0,method='SLSQP',bounds=bounds,constraints=cons)
    if not res.success:
        raise ValueError(f'Optimización de máximo Sharpe no convergió:{res.message}')
    w_opt = np.array(res.x,dtype=float)
    w_opt = w_opt/w_opt.sum()
    return w_opt
#--------------- 

#---------------Función que encuentra el portafolio de rendimiento objetivo
def port_rend_objetivo(media_anual,cov_anual,rend_objetivo):
    n = len(media_anual)
    x0 = np.repeat(1/n,n)
    bounds = tuple((0.0,1.0) for _ in range(n))
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w)-1},
            {'type':'eq', 'fun': lambda w: np.dot(w,media_anual)-rend_objetivo},)
    def obj(w):
        w = np.array(w,dtype=float)
        return float(w @ cov_anual @ w)
    res = minimize(obj,x0,method='SLSQP',bounds=bounds,constraints=cons)
    if not res.success:
        raise ValueError(f'Optimización de rendimiento objetivo no convergió:{res.message}')
    w_opt = np.array(res.x,dtype=float)
    w_opt = w_opt/w_opt.sum()
    return w_opt
#--------------- 