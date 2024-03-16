import pandas as pd
from scipy import stats
import numpy as np

# Carregar a base de dados usando ';' como separador
df = pd.read_csv('Casos_e_obitos_ESP.csv', sep=';')

# Número de registros
num_registros = len(df)
print("Número total de registros:", num_registros)

# Valores faltando por coluna
valores_faltando = df.isnull().sum()
print("\nValores faltando por coluna:")
print(valores_faltando)

# Número de linhas duplicadas
linhas_duplicadas = df.duplicated().sum()
print("\nNúmero de linhas duplicadas:", linhas_duplicadas)

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
estatisticas_descritivas = df.describe()
print(estatisticas_descritivas)

# Outliers (exemplo usando Z-score para encontrar outliers na coluna 'valor')
z_scores = np.abs(stats.zscore(df['Idade']))
threshold = 1
outliers = df[(z_scores > threshold)]
print("\nNúmero de outliers na coluna 'Idade':", len(outliers))
print(outliers)
