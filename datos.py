import numpy as np
import pandas as pd
import yfinance as yf
from config import DIAS_HAB

#---------------
#Este módulo se encargará de la descarga de datos necesaria de yfinance, calcula 
#rendimientos y obtiene medias y covarianzas.
#---------------

#--------------- Funcion para extraer precios
def _extraer_precios(df_yf: pd.DataFrame) -> pd.DataFrame:
    if df_yf is None or df_yf.empty:
        raise ValueError('El DataFrame está vacío')
    
    #Caso 1: Varios niveles en yfinance
    if isinstance(df_yf.columns, pd.MultiIndex):
        num_niveles = df_yf.columns.nlevels
        n_encontrado = None
        close_encontrado = None
        
        for nivel in range(num_niveles):
            valores_niveles = set(df_yf.columns.get_level_values(nivel))
            
            
            if 'Adj Close' in valores_niveles:
                n_encontrado = nivel
                close_encontrado = 'Adj Close'
                break
            if 'Close' in valores_niveles:
                n_encontrado = nivel
                close_encontrado = 'Close'
                break
            
        if n_encontrado is None:
            raise ValueError('No se encontraron los valores de Cierre.')
        
        precios = df_yf.xs(close_encontrado, axis=1, level=n_encontrado)
    #Caso 2: Solo un nivel en yfinance    
    else:
        columnas = list(df_yf.columns)
        if 'Adj Close' in columnas:
            precios = df_yf['Adj Close']
        elif 'Close' in columnas:
            precios = df_yf['Close']
        else: 
            raise ValueError('No se encontraron los valores de Cierre')
    
    if isinstance(precios, pd.Series):
        precios = precios.to_frame()
    precios = precios.dropna(how='all')
    
    return precios
#---------------

#---------------Función de descarga de datos 
def download_precios(lista_etfs, fecha_inicio, fecha_fin) -> pd.DataFrame:
    raw_data = yf.download(tickers=lista_etfs, start=fecha_inicio, end=fecha_fin, auto_adjust=False, progress=False,)
    precios = _extraer_precios(raw_data)
    columnas_disponibles = [c for c in precios.columns if c in lista_etfs]
    precios = precios[columnas_disponibles]
    
    return precios
#---------------

#---------------Función que calcula rendimientos
def calcular_rendimientos(precios: pd.DataFrame,tipo: str= 'log') -> pd.DataFrame:
    if precios is None or precios.empty:
        raise ValueError('El DataFrame de Precios está vacío')
    if tipo == 'log':
        rend = np.log(precios/precios.shift(1))
    elif tipo == 'simple':
        rend = precios.pct_change()
    else:
        raise ValueError('El tipo de rendimiento es desconocido. Usa log o simple.')
    
    rend = rend.dropna(how='all')
    
    return rend
#---------------

#---------------Función que calcula medias y covarianzas anuales
def estadis_anuales(rendimientos: pd.DataFrame, dias_ano: int = DIAS_HAB):
    if rendimientos is None or rendimientos.empty:
        raise ValueError('El DataFrame de rendimientos está vacío.')
    
    #---------------Media y cov diaria
    media_diaria = rendimientos.mean()
    cov_diaria = rendimientos.cov()
    #---------------
    
    #---------------Media y cov anual
    media_anual = media_diaria * dias_ano
    cov_anual = cov_diaria * dias_ano
    #---------------
    return media_anual, cov_anual
#---------------
    
        

    

        
        
                
            
        
        

