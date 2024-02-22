import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import ttest_ind, chi2_contingency
from scipy.stats import f_oneway

'''Función 1 - Pedro'''
def describe_df(df):
    """
    Genera estadísticas descriptivas para un DataFrame.

    Argumentos:
    df (DataFrame): El DataFrame para el cual se generan las estadísticas.

    Retorna:
    DataFrame: Un DataFrame que contiene estadísticas descriptivas transpuestas.
    """

    # Creamos un DataFrame vacío para almacenar las estadísticas descriptivas
    result = pd.DataFrame()

    # Agregamos la columna de tipos de datos
    result["DATA_TYPE"] = df.dtypes
    
    # Calculamos el porcentaje de valores faltantes en cada columna
    result["MISSING (%)"] = (df.isnull().sum() / len(df)) * 100
    
    # Calculamos la cantidad de valores únicos en cada columna
    result["UNIQUE_VALUES"] = df.nunique()
    
    # Calculamos el porcentaje de cardinalidad en cada columna
    result["CARDIN (%)"] = df.nunique() / len(df) * 100

    # Devolvemos el DataFrame transpuesto para tener las estadísticas en filas
    return result.T



'''Función 2 - Pedro'''
def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Sugiere tipos de variables para las columnas de un DataFrame.

    Argumentos:
    df (DataFrame): El DataFrame para el cual se sugieren los tipos de variables.
    umbral_categoria (int): Umbral para determinar si una columna es categórica.
    umbral_continua (int): Umbral para determinar si una columna es continua.

    Retorna:
    DataFrame: Un DataFrame que contiene sugerencias de tipos de variables para cada columna.
    """

    # Obtenemos estadísticas descriptivas usando la función describe_df
    card_df = describe_df(df)
    
    # Creamos un DataFrame vacío para almacenar las sugerencias de tipos de variables
    result = pd.DataFrame()
    
    # Agregamos la columna de nombres de variables
    result["nombre_variable"] = df.columns
    # Agregamos la columna de tipos de variables con valores iniciales vacíos
    result["tipo_sugerido"] = ""

    # Iteramos a través de las estadísticas descriptivas
    for i, value in enumerate(card_df.loc["UNIQUE_VALUES"]):
        if value == 2:
            # Si la columna tiene 2 valores únicos, se sugiere como "binaria"
            result.loc[i, "tipo_sugerido"] = "binario"
        elif value < umbral_categoria:
            # Si la columna tiene menos valores únicos que el umbral de categoría, se sugiere como "categórica"
            result.loc[i, "tipo_sugerido"] = "categorica"
        elif value >= umbral_categoria:
            # Si la columna tiene igual o más valores únicos que el umbral de categoría, se evalúa la cardinalidad
            for j, values in enumerate(card_df.loc["CARDIN (%)"]):
                if result.loc[j, "tipo_sugerido"] != "":
                    # Si la sugerencia ya fue asignada, se omite
                    continue
                if values > umbral_continua:
                    # Si la cardinalidad supera el umbral, se sugiere como "numerica continua"
                    result.loc[j, "tipo_sugerido"] = "numerica continua"
                else:
                    # Si la cardinalidad no supera el umbral, se sugiere como "numerica discreta"
                    result.loc[j, "tipo_sugerido"] = "numerica discreta"

    return result


'''Función 3 - Pedro'''
def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Obtiene las características numéricas que están correlacionadas con la variable objetivo para regresión numerica.

    Argumentos:
    df (DataFrame): El DataFrame que contiene las características y la variable objetivo.
    target_col (str): El nombre de la columna que representa la variable objetivo.
    umbral_corr (float): Umbral de correlación para considerar una característica relevante.
    pvalue (float, opcional): Umbral p-value para la significancia estadística. Si es None, no se aplica.

    Retorna:
    list: Lista de nombres de características numéricas correlacionadas con la variable objetivo.
    """
    # Verificamos si la columna objetivo es válida
    if target_col not in df.columns:
        print(f"Error: {target_col} no es una columna válida.")
        return None
    
    # Verificamos si el umbral de correlación está en el rango válido [0, 1]
    if not (0 <= umbral_corr <= 1):
        print("Error: umbral_corr tiene que estar entre 0 y 1.")
        return None
    
    # Verificamos si el p-value (si se proporciona) está en el rango válido [0, 1]
    if pvalue is not None and not (0 <= pvalue <= 1):
        print("Error: pvalue tiene que estar entre 0 y 1.")
        return None

    # Seleccionamos las columnas numéricas del DataFrame
    num_cols = df.select_dtypes(include=['number']).columns
    corr_cols = []

    # Iteramos a través de las columnas numéricas para calcular la correlación con la variable objetivo
    for col in num_cols:
        if col != target_col:
            correlation, p_value = pearsonr(df[target_col], df[col])

            # Verificamos si la correlación es mayor al umbral y si el p-value (si se proporciona) es aceptable
            if abs(correlation) > umbral_corr and (pvalue is None or p_value <= (1 - pvalue)):
                corr_cols.append(col)
    
    return corr_cols




'''Función 4 -  Jaime'''
def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0.00, pvalue=0.05):
    """
    Genera pairplots para visualizar las relaciones entre la columna target y otras columnas numéricas del DF,
    filtrando basado en el umbral de correlación y el valor p. También maneja situaciones donde ninguna característica
    cumple con los criterios establecidos, mostrando un mensaje en tal caso.

    Argumentos:
    df: El DF que contiene los datos.

    target_col: El nombre de la columna target en el DataFrame. Por defecto es una cadena vacía.

    columns (opcional): Una lista de nombres de columnas features con la que realizar el análisis. Si está vacía, se seleccionarán
                               automáticamente todas las variables numéricas del DF. Por defecto es una lista vacía.
    umbral_corr (opcional): El valor mínimo de correlación que debe existir entre el target y las features. Por defecto es 0.2
                                    
    pvalue (opcional): El valor p utilizado para filtrar las columnas determinar si existe  significancia estadística entre el target y las features. Por defecto es 0.05 (En None no se aplica)
                              

    Retorna:
    list: Una lista de nombres de columnas que cumplen con los criterios de correlación y significancia estadística establecidos.
          Devuelve una lista vacía si ninguna característica cumple con los criterios.
    """
    
    # Comprobación de la validez de 'target_col' (comprobar si es numérica)
    if target_col not in df.columns or not np.issubdtype(df[target_col].dtype, np.number): # se comprueba si existe una columna en el df que tenga el mismo nombre que en target_col y np.issubdtype comprueba el tipo de datos de la columna
        # si alguna de las dos condiciones no se cumple se imprime un mensaje de que la columnas añadida en el parámetro de la función no es válido
        print(f"El 'target_col' especificado ({target_col}) no es una columna numérica del dataframe.")
        return None

    # Si 'columns' está vacío, seleccionar todas las variables numéricas del dataframe
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist() # selección de columnas que pertenezcan al tipo np.number y los añade a una lista
        if target_col in columns:
            columns.remove(target_col) # si se detecta la variable target se quita de la lista
    
    # Anaísis de correlaciones y selección de columnas
    selected_columns = []  # lista vacía para almacenar las columnas que cumplen los criterios
    for col in columns:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number): # se comprueba que la columna exista y que tenga un dtype numérico
            # se calcula la correlación usando el método .corr() 
            corr = df[target_col].corr(df[col])
            # primero se aplica el filtro de umbral de correlación
            if abs(corr) >= umbral_corr: # se verifica que el valor otenido sea igual o mayor al añadido en el parámetro umbral_corr
                if pvalue is not None: # se comprueba si se ha añadido un p value en el parámetro de pvalue. si es None entonces no se aplica
                    # Cálculo del valor p usando pearsonr
                    _, p_val = pearsonr(df[target_col], df[col]) # se ignora el coeficiente de correlacion "_" el primer valor, ya que solo quiero el p value
                    if p_val < pvalue:
                        selected_columns.append(col) # si el p value obtenido es menor qu el especificado en el parámetro de pvalue se añade a la lista de 'selected_columns' 
                else:
                    # Si pvalue es None, solo consideramos el umbral de correlación
                    selected_columns.append(col)

    # Si ninguna columna cumple los criterios
    if not selected_columns:
        print("Ninguna característica cumple con los criterios establecidos.")
        return []

    # Dividir las columnas seleccionadas en grupos para los pairplots (GPT)
    grouped_columns = [selected_columns[i:i + 4] for i in range(0, len(selected_columns), 4)]
    #range(0, len(selected_columns), 4) genera una secuencia de números que comienza en 0, termina en la longitud de selected_columns, y avanza en pasos de 4. Estos números se utilizan como índices de inicio para cada sublista.  
    for group in grouped_columns:
        # Generar y mostrar el pairplot para cada grupo de columnas
        sns.pairplot(df, vars=[target_col] + group)
        plt.show()

    return selected_columns



'''Función 5 - Adri'''
def test_relacion_categoricas(df, target_col, pvalue=0.05):
    """
    Realiza pruebas estadísticas para evaluar la relación entre una columna numérica continua
    (target_col) y columnas categóricas en un DataFrame.

    Parámetros:
    - df: DataFrame, el conjunto de datos.
    - target_col: str, nombre de la columna objetivo (debe ser numérica continua).
    - pvalue: float, umbral de valor p para determinar la significancia estadística.

    Retorna:
    - Lista de nombres de columnas categóricas con relaciones estadísticamente significativas.

    Si hay errores o no se encuentran relaciones significativas, se imprime un mensaje y retorna None.
    """
    lista = []      # Lista para almacenar resultados intermedios
    lista2 = []     # Lista para almacenar columnas con relaciones estadísticamente significativas

    # Comprobar si 'target_col' es una columna numérica continua
    if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]): 
        '''Verifica dos condiciones: Si target_col no está presente en las columnas del 
    DataFrame (df.columns) y Si el tipo de datos de la columna target_col no es numérico continuo'''
        print(f"Error: '{target_col}' no es una columna numérica continua válida.")
        return None
    
    # Filtrar columnas categóricas
    col_max_cardinalidad = df.select_dtypes(include=["object", "category"]).columns #especifica que solo se deben incluir las columnas con tipos de datos 'object' (cadenas de texto) o 'category' (columnas categóricas).
    
    # Comprobar si hay columnas categóricas
    if len(col_max_cardinalidad) == 0:
        print("Error: No hay columnas categóricas en el dataframe.")
        return None
    
    # Iterar sobre las columnas categóricas con mayor cardinalidad
    for col in col_max_cardinalidad:
        # Verificar si la columna es numérica
        if pd.api.types.is_numeric_dtype(df[col]): #Si la columna es numérica, realiza la prueba t de independencia (ttest_ind)
            # Realizar la prueba t de independencia
            test_result = ttest_ind(df[col], df[target_col]) #La prueba t evalúa si hay diferencias significativas entre las medias de dos grupos (en este caso, las muestras de la columna categórica y la columna objetivo)
            lista.append((col, test_result))  # Agregar el resultado a la lista
        else: #Si la columna no es numérica, significa que es categórica.
            # Crear una tabla de contingencia y realizar el test de chi-cuadrado
            contingency_table = pd.crosstab(df[col], df[target_col])#crea una tabla de contingencia, tabla que muestra la distribución conjunta de las frecuencias de dos o más variables categóricas.
            test_result = chi2_contingency(contingency_table) #El test de chi-cuadrado evalúa si existe una asociación significativa entre las variables categóricas.
            lista.append((col, test_result))  # Agregar el resultado a la lista
    
    # Evaluar el p-value del test de relación
    for col, result in lista:
        if result.pvalue < pvalue:
            lista2.append(col)  # Agregar la columna con relación significativa a la lista2
    
    # Imprimir mensaje si no se encontraron relaciones significativas
    if not lista2:
        print(f"No se encontraron relaciones estadísticamente significativas para '{target_col}'.")
    
    # Devolver la lista de columnas con relaciones significativas
    return lista2



'''Función 6 - Nur'''

def plot_features_cat_regression(df, target = "", columns = [], p_value = .05):

    """
    Pinta la distribución del target por categoría en cada columna,
    después de haber realizado el test de relación ANOVA.

    Argumentos:
        df (DataFrame): el dataset que estamos estudiando para el modelado
        target (string): la columna que hemos elegido como "y" del modelado
        columns (list): lista de las columnas categóricas del dataset
        p_value (float): un número entre 0 y 1 que hemos elegido como p_value
                        de la probabilidad de que target y col estén relacionadas


    Retorna:
    Una lista con las columnas que tienen una relación significativa con el target
    """

    # Verificar si df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El argumento df tiene que ser un DataFrame")
    
    # Verificar si target es un str
    if not isinstance(target, str):
        raise ValueError("El argumento target tiene que ser un string")
    
    # Verificar si target es una columna numérica
    if target not in df.select_dtypes(include=['float', 'int']).columns:
        raise ValueError("El argumento target tiene que ser una columna numérica")
    
    # Verificar si columns es una lista
    if not isinstance(columns, list):
        raise ValueError("El argumento columns debe ser una lista")

    # verificar que todos los elementos de la lista sean columnas en df
    valid_columns = df.columns
    for col in columns:
        if col not in valid_columns:
            raise ValueError(f"La columna {col} en 'columns' no es una columna válida en el DataFrame")
    
    
    # Verificar si p_value está entre 0 y 1
    if p_value < 0 or p_value > 1:
        raise ValueError("El argumento p_value debe ser un número entre 0 y 1")

    # si la lista columns está vacía le metemos las categóricas
    if not columns:
        columns = df.select_dtypes(exclude=['number']).columns.tolist()
        
    # test de relacion anova
    col_significativas = []
    for columna in columns:
        grupos = [df[df[columna] == valor][target] for valor in df[columna].unique() if isinstance(valor, str)] # agrupamos los valores de target por las categorias col
        if len(grupos) > 1:
            p_valor = f_oneway(*grupos).pvalue
            if p_valor < p_value:
                col_significativas.append(columna) # si p_valor es inferior al p_value metemos la columna en la lista para pintar

    if not col_significativas:
        print("Parece que ninguna columna tiene mucha relación con target")
        return None


    # Para pintar los subplots agrupados (3 por fila)
    num_plots = len(col_significativas)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    for i, col in enumerate(col_significativas):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.histplot(data=df, x=target, hue=col, bins=50, kde=True)

    # Ocultar ejes sobrantes
    for j in range(num_plots, num_rows * num_cols):
        plt.subplot(num_rows, num_cols, j + 1).set_axis_off()

    plt.tight_layout()
    plt.show()

    return col_significativas




