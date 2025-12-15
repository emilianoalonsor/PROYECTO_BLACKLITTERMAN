import numpy as np


#Este módulo se encarga de definir el "universo" de inversión, define parámetros generales y los pesos
#de benchmark correspondientes al universo elegido

#---------------DIAS HABILES
DIAS_HAB = 252
#---------------

#---------------TASA LIBRE DE RIESGO INICIAL
TINI_RIESGO = 0.04
#---------------

#---------------UNIVERSOS DE INVERSION
PESO_REGIONES = {
    'SPLG' : 70.62,
    'EWC' : 3.23,
    'IEUR' : 11.76,
    'EEM' : 9.02,
    'EWJ' : 5.37,
}

PESO_SECTORES = {
    'XLC' : 9.99,
    'XLY' : 10.25,
    'XLP' : 4.82,
    'XLE' : 2.95,
    'XLF' : 13.07,
    'XLV' : 9.58,
    'XLI' : 8.09,
    'XLB' : 1.66,
    'XLRE' : 1.87,
    'XLK' : 35.35,
    'XLU' : 2.37,
}
#---------------

#--------------- FUNCION QUE ASIGNA PESOS SEGUN UNIVERSO ELEGIDO
def obtener_universo(estrategia: str):
    if estrategia == 'Regiones':
        dic_pesos = PESO_REGIONES
    elif estrategia == 'Sectores EUA':
        dic_pesos = PESO_SECTORES
    else:
        raise ValueError('Estrategia no reconocida. Escoge una de las dos opciones válidas.')

    ETFS = list(dic_pesos.keys())
    pesos_num = np.array(list(dic_pesos.values()), dtype= float)
    pesos_benchmark = pesos_num / pesos_num.sum()
   
    return ETFS, pesos_benchmark

#---------------
    
 