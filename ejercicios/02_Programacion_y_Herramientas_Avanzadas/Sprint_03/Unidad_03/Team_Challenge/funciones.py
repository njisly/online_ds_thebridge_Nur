import numpy as np

def coloca_barco(tablero, barco):
    for pieza in barco:
        tablero[pieza] = "O"

def crea_tablero(dimensiones):
    return np.full((dimensiones))

TAM_DEFECTO = 10
