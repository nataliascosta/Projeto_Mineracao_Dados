import pandas as pd
import re

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

# Função para verificar o padrão da data
def verificar_padrao_data(data):
    if pd.isnull(data):  # Verificar se o valor é nulo
        return False  # Se for nulo, não é uma data suja
    # Verificar se a data possui 8 dígitos e o formato xx/xx/xxxx
    padrao = re.compile(r'^\d{2}/\d{2}/\d{4}$')
    if re.match(padrao, str(data)):  # Convertendo para string antes da verificação
        return True
    else:
        return False
    
# Verificar se a data é suja
datas_sujas = []
for index, row in df.iterrows():
    if not verificar_padrao_data(row['Data_Inicio_Sintomas']):
        datas_sujas.append((index, row['Data_Inicio_Sintomas']))

# Imprimir as datas sujas
print("Datas sujas:")
for index, data in datas_sujas:
    print("Linha:", index, "- Data:", data)

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
estatisticas_descritivas = df.describe()
print(estatisticas_descritivas)

# Calcular o IQR
Q1 = df['Idade'].quantile(0.25)
Q3 = df['Idade'].quantile(0.75)
IQR = Q3 - Q1

# Identificar outliers
outliers = df[(df['Idade'] < Q1 - 1.5 * IQR) | (df['Idade'] > Q3 + 1.5 * IQR)]

# Imprimir resultados
print("Número de outliers na coluna 'Idade':", len(outliers))
print(outliers)

# Salvar os outliers em um arquivo CSV
outliers.to_csv('outliers.csv', index=False)