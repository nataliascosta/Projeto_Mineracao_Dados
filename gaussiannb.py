from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd


# Ler o arquivo CSV
dados = pd.read_csv('Casos_e_obitos_ESP.csv', sep=';')

# Remover linhas com valores faltantes
dados.dropna(inplace=True)

# Selecionar features e variável alvo
X = dados[['Asma', 'Cardiopatia', 'Data_Inicio_Sintomas', 'Diabetes', 'Diagnostico_Covid19', 'Doenca_Hematologica',
           'Doenca_Hepatica', 'Doenca_Neurologica', 'Doenca Renal', 'Genero', 'Idade', 'Imunodepressao', 'Municipio',
           'Obesidade', 'Outros_Fatores_De_Risco', 'Pneumopatia', 'Puérpera', 'Síndrome_De_Down']].copy()
y = dados['Obito'].copy()


# Inicializar o codificador de rótulos
le = LabelEncoder()

# Codificar variáveis categóricas
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = le.fit_transform(X[column].astype(str))

# Inicializar o escalador Min-Max
scaler = MinMaxScaler()

# Normalizar os dados
X_normalizado = scaler.fit_transform(X)

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciar o classificador Gaussian Naive Bayes
clf = GaussianNB()

# Treinar o classificador com o conjunto de treinamento
clf.fit(X_train, y_train)

# Realizar previsões no conjunto de teste
y_pred = clf.predict(X_test)

# Exibir o relatório de classificação
report_dict = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(report_dict).transpose()
df.to_csv('classification_report.csv', index=True)

print("Relatório de classificação exportado como 'classification_report.csv'.")