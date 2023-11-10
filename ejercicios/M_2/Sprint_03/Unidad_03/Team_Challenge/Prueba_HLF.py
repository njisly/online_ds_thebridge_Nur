#IMPORTS
# VARIABLES Y CONSTANTES GLOBALES
# DEFINICION DE FUNCIONES
# DEFINICION DE CLASES
# SI LO NECESITAS, CODIGO PRINCIPAL
# el programa tiene que estar en main.py
# las funciones tienen que estar en otro fichero pq nos vamos a partir las funciones
# desde el fichero main tenemos que importarnos los valores
# (otro archivo en el mismo directorio que main)

# form sys import path
# path.append(".\\")

# me puedo ejecutar funciones para ver si me esta funcionando la libreria
# 
if __name__ == "__main__"
    print(crea_tablero(10,15))
# esto hace que el codigo que viene despues si yo le pido llamar a funciones
# puedo probar codigo ejecutando mi libreria sin molestar a quien quiera seguir llamando cosas

import numpy as np

# from funciones import TAM_DEFECTO, crea_tablero, coloca_barco # ya les puedo quitar el alias fn
# from funciones import * y te lo trae todo como si estuviera en mi archivo

def crea_tablero(dimensiones):
    return np.full(dimensiones, " ")

tam_tablero = input(f"Que tamaño quieres? (ancho, alto) {fn.TAM_DEFECTO, fn.TAM_DEFECTO}")

if tam_tablero != "":
    lista_dimensiones = [int(elemento) for elemento in tam_tablero.split(",")]
else:
    lista_dimensiones = (fn.TAM_DEFECTO, fn.TAM_DEFECTO)


lista_dimensiones = [int(elemento) for elemento in tam_tablero.split(",")]

tablero = fn.crea_tablero(lista_dimensiones)

fn.coloca_barco(tablero, [(1,1), (1,2), (1,3)])

print(tablero)


adrian f jaime pedro