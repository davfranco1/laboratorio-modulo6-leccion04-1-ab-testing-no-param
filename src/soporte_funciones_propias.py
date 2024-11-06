import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools


def iteraciones_normalidad(df, columna_grupo, columna_metrica, iteraciones=5, alpha=0.05):
    tamanios_grupos1 = list(df[columna_grupo].value_counts())
    grupo_mas_peq = float(min(tamanios_grupos1))
    grupos = list(df[columna_grupo].unique())
    
    conteo_normalidad = {grupo: 0 for grupo in grupos}
    conteo_no_normalidad = {grupo: 0 for grupo in grupos}

    for i in range(iteraciones):
        df_sampled = df.sample(int(grupo_mas_peq))
        df_sampled = df_sampled[[columna_grupo, columna_metrica]]

        if grupo_mas_peq < 30:
            for valor in grupos:
                stats_result = stats.shapiro(df_sampled[df_sampled[columna_grupo] == valor][columna_metrica])
                pvalor = stats_result.pvalue
                if pvalor > alpha:
                    conteo_normalidad[valor] += 1
                else:
                    conteo_no_normalidad[valor] += 1
        else:
            datos = df.groupby(columna_grupo)[columna_metrica].agg(["mean", "std"])

            for grupo in grupos:
                dist = np.random.normal(loc=datos.loc[grupo, "mean"], scale=datos.loc[grupo, "std"], size=df.shape[0])
                _, pvalor = stats.kstest(df[df[columna_grupo] == grupo][columna_metrica], dist)
                
                if pvalor > alpha:
                    conteo_normalidad[grupo] += 1
                else:
                    conteo_no_normalidad[grupo] += 1
    
    for grupo in grupos:
        print(f"Para el grupo {grupo}: {conteo_normalidad[grupo]} p-valores mayores a {alpha} que siguen una distribución normal, y {conteo_no_normalidad[grupo]} p-valores menores a {alpha}, por lo que no siguen una distribución normal.")


def histogramas_homocedasticidad(df, columna_grupo, columna_metrica):
    # Obtener los tamaños de los grupos
    tamanios_grupos = df[columna_grupo].nunique()
    # Calcular el número de filas y columnas para los subplots
    filas = (tamanios_grupos // 2) + (tamanios_grupos % 2)
    columnas = 2
    
    # Crear subplots de acuerdo al tamaño de los grupos
    fig, axes = plt.subplots(filas, columnas, figsize=(10, 5))
    axes = axes.flatten()

    lista_grupos = list(df[columna_grupo].unique())

    # Crear histogramas para cada grupo
    for indice, grupo in enumerate(lista_grupos):
        sns.histplot(df[df[columna_grupo] == grupo], x=columna_metrica, ax=axes[indice], kde=True)
        axes[indice].set_xlabel(f'Grupo {grupo}')

    # Eliminar subplots vacíos si hay un número impar de grupos
    if tamanios_grupos % 2 != 0:
        fig.delaxes(axes[-1])
        
    plt.tight_layout() 
    plt.show()


def test_homocedasticidad(df, columna_grupos, columna_metrica, test):
    test = test.lower()
    unicos = df[columna_grupos].unique()

    for grupo in unicos:
        df_metrica = df[df[columna_grupos] == grupo][columna_metrica]
        globals()[grupo] = df_metrica
    
    if test == "levene":
        print(stats.levene(*[globals()[var] for var in unicos]))

    else:
        test == "bartlett"
        print(stats.bartlett(*[globals()[var] for var in unicos]))


def crear_df_grupos (df, col_grupo, col_metrica):

    for valor in df[col_grupo].unique():
        globals()[valor.lower()] = df[df[col_grupo]==valor][col_metrica]

    return list(df[col_grupo].unique())


def elegir_test(valores, dependencia=False):

    if len(valores) > 2:
        print("Test Kruskal")
        _, pvalor = stats.kruskal(*[globals()[var.lower()] for var in valores])

    elif len(valores)==2 and dependencia==True:
        print("Test Wilcoxon")
        _, pvalor = stats.wilcoxon(*[globals()[var.lower()] for var in valores])

    elif len(valores)==2 and dependencia==False:
        print("Test Mann-Whitney")
        _, pvalor = stats.mannwhitneyu(*[globals()[var.lower()] for var in valores])

    else:
        print("Valores no válidos")
        return
    
    if pvalor < 0.05:
        print(f"Siendo el p-valor {pvalor}, existen diferencias significativas entre los grupos.")
    else:
        print(f"Siendo el p-valor {round(pvalor,4)}, no existen diferencias significativas entre los grupos.")


def combinaciones_test(df, columna_grupo, columna_metrica):
    """
    Genera combinaciones de valores únicos en una columna del dataframe y realiza un test estadístico.

    Parámetros:
    - df: DataFrame.
    - columna_grupo: Nombre de la columna para agrupar.
    - columna_metrica: Nombre de la columna métrica.
    - test_function: Función de test a aplicar a cada combinación.

    Retorna:
    - None. Imprime los resultados del test.
    """
    # Generar combinaciones de 2 en 2 de los valores únicos en columna_grupo
    lista_combinaciones = list(itertools.combinations(df[columna_grupo].unique(), 2))

    # Iterar sobre cada combinación y realizar el test
    for combinacion in lista_combinaciones:
        lista_valores = []
        print("\n")
        for valor in combinacion:
            globals()[valor.lower()] = df[df[columna_grupo] == valor][columna_metrica].values  # Guardar los valores en variables globales
            lista_valores.append(valor)
        print(f"Para los grupos {lista_valores}")
        elegir_test(lista_valores)


def hisplot_grupos(df, columna_grupo, columna_metrica):
    
    lista_grupos = list(df[columna_grupo].unique())
    if len(lista_grupos) % 2 == 0:
        filas = len(lista_grupos) // 2
        pares=True
    else:
        filas = len(lista_grupos) // 2 +1
        pares = False

    fig, axes = plt.subplots(filas,2, figsize=(15,10))
    axes= axes.flat

    for i,grupo in enumerate(lista_grupos):
        sns.histplot(df[df[columna_grupo]==grupo], x =columna_metrica, kde=True, ax=axes[i])
        axes[i].set_xlabel(grupo)
        axes[i].set_ylabel("Count")

    if not pares:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()