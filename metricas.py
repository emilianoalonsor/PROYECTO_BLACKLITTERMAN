import numpy as np
import pandas as pd
from config import DIAS_HAB

#---------------Explicación del módulo
'''
Este módulo se encarga de calcular:
-Rendimientos dado un vector de pesos
-Máximo drawdawn
-VaR y CVaR
-Sharpe y Sortino
-Devolver todas las métricas en un diccionario
'''
#---------------

#---------------Función que calcula los rendimientos del portafolio 
def rend_portafolio(rendimientos: pd.DataFrame, pesos: np.ndarray) -> pd.Series:
    if rendimientos is None or rendimientos.empty:
        raise ValueError('El DataFrame de rendimientos está vacío.')
    pesos = np.array(pesos, dtype=float)
    if pesos.ndim !=1:
        raise ValueError('Los pesos deben ser un arreglo unidimensional.')
    suma_pesos = pesos.sum()
    if suma_pesos == 0:
        raise ValueError('La suma de pesos es 0.')
    pesos_norm = pesos/suma_pesos
    serie_portafolio = rendimientos.dot(pesos_norm)
    return serie_portafolio
#---------------

#---------------Función que calcula el maximo drawddown 
def m_drawdown(rend_port: pd.Series) -> float:
    if rend_port is None or rend_port.empty:
        raise ValueError('La serie de rendimientos es vacía.')
    valor_acumulado = (1+ rend_port).cumprod()
    max_hist = valor_acumulado.cummax()
    drawdowns = (valor_acumulado - max_hist)/max_hist
    return drawdowns.min()
#---------------

#---------------Funcion que calcula el VaR y CVaR
def var_cvar(rend_port: pd.Series, niv_confianza: float = 0.95):
    if rend_port is None or rend_port.empty:
        raise ValueError('La serie de rendimientos del portafolio está vacía.')
    if not (0.0 < niv_confianza < 1.0):
        raise ValueError('El nivel de confianza es inválido.')
    rend_ordenados = np.sort(rend_port.values)
    indice_var = int((1- niv_confianza)*len(rend_ordenados))
    var_diario = -rend_ordenados[indice_var]
    cvar_diario = -rend_ordenados[:indice_var + 1].mean()
    return var_diario, cvar_diario
#---------------

#---------------Función que calcula la Beta de un portafolio respecto a un Benchmark
def beta_vs_bench(rend_port: pd.Series, rend_bench: pd.Series) -> float:
    
    if rend_port is None or rend_port.empty:
        raise ValueError('La serie de rendimientos del portafolio definido está vacía')
    if rend_bench is None or rend_bench.empty:
        raise ValueError('La serie de rendimientos del benchmark está vacía')
    datos = pd.concat([rend_port,rend_bench], axis=1).dropna()
    if datos.shape[0] < 2:
        return np.nan
    cov_matriz = np.cov(datos.iloc[:,0],datos.iloc[:,1])
    cov_pyb = cov_matriz[0,1]
    var_b = cov_matriz[1,1]
    if var_b ==0:
        return np.nan
    return cov_pyb/var_b
#---------------     

#---------------Función que calcula el rendimiento anual esperado, la volatilidad y el sharpe.
def sharpe_y_vol(rend_port: pd.Series, tasa_libre_anual: float, dias_ano: int=DIAS_HAB):
    if rend_port is None or rend_port.empty:
        raise ValueError('La serie de rendimientos del portafolio está vacía.')
    media_diaria = rend_port.mean()
    desv_diaria = rend_port.std()
    media_anual = media_diaria * dias_ano
    desv_anual = desv_diaria * np.sqrt(dias_ano)
    if desv_anual > 0 :
        sharpe = (media_anual - tasa_libre_anual)/desv_anual
    else:
        sharpe = np.nan
    return media_anual,desv_anual,sharpe
#---------------

#---------------Función que calcula el indice de Sortino 
def sortino_anual(rend_port: pd.Series, tasa_libre_anual:float, dias_ano: int = DIAS_HAB):
    if rend_port is None or rend_port.empty:
        raise ValueError('La serie de rendimientos del portafolio está vacía.')
    tasa_objetivo_d = tasa_libre_anual/dias_ano
    rend_exceso = rend_port - tasa_objetivo_d
    perdidas = np.minimum(rend_exceso, 0.0)
    
    desv_d_perdidas = np.sqrt((perdidas**2).mean())
    desv_a_perdidas = desv_d_perdidas * np.sqrt(dias_ano)
    media_diaria = rend_port.mean()
    media_anual = media_diaria * dias_ano
    if desv_a_perdidas > 0:
        sortino = (media_anual-tasa_libre_anual)/desv_a_perdidas
    else:
        sortino = np.nan
    return sortino
#---------------

#---------------Función que regresa un diccionario con todas las métricas
def resumen_metricas(
    rendimientos:pd.DataFrame, 
    pesos_port: np.ndarray,
    pesos_bench: np.ndarray,
    tasa_libre_anual:float,
    nivel_var:float = 0.95,
    dias_ano:int = DIAS_HAB,
):
    rend_port = rend_portafolio(rendimientos, pesos_port)
    rend_bench = rend_portafolio(rendimientos,pesos_bench)
    mu_anual, desv_anual, sharpe = sharpe_y_vol(rend_port, tasa_libre_anual,dias_ano)
    sortino = sortino_anual(rend_port,tasa_libre_anual,dias_ano)
    beta = beta_vs_bench(rend_port,rend_bench)
    max_drawdown = m_drawdown(rend_port)
    var_diario, cvar_diario = var_cvar(rend_port,niv_confianza=nivel_var)
    var_anual = var_diario *np.sqrt(dias_ano)
    cvar_anual = cvar_diario * np.sqrt(dias_ano)
    sesgo = rend_port.skew()
    curt = rend_port.kurtosis()
    
    metricas = {
        'Rendimiento anual esperado': mu_anual,
        'Volatilidad anual': desv_anual,
        'Sharpe': sharpe,
        'Sortino':sortino,
        'Beta vs Benchmark': beta,
        'Max Drawdown':max_drawdown,
        f'Var anual histórico({int(nivel_var*100)}%)':var_anual,
        f'CVaR anual histórico({int(nivel_var*100)}%)':cvar_anual,
        'Sesgo':sesgo,
        'Curtosis':curt,
    }    
    return metricas, rend_port,rend_bench
#---------------
