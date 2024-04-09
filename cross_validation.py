from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

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


# Lista dos classificadores
classifiers = [
    GaussianNB(),  # Presta
    KNeighborsClassifier(),  # Presta
    LogisticRegression(),  # Presta
    XGBClassifier()  # Presta
]

# Dicionário para armazenar as métricas de cada classificador
metrics = {
    'Acurácia': [],
    'Precisão': [],
    'Sensibilidade/Recall': [],
    'Medida-F1': []
}

# Executar validação cruzada para cada classificador
for classifier in classifiers:
    # Calcula a acurácia usando validação cruzada (cv=10 para 10 folds)
    accuracy = cross_val_score(classifier, X_normalizado, y, cv=10, scoring='accuracy').mean()
    # Calcula a precisão usando validação cruzada
    precision = cross_val_score(classifier, X_normalizado, y, cv=10, scoring='precision').mean()
    # Calcula a sensibilidade/recall usando validação cruzada
    recall = cross_val_score(classifier, X_normalizado, y, cv=10, scoring='recall').mean()
    # Calcula a medida-F1 usando validação cruzada
    f1 = cross_val_score(classifier, X_normalizado, y, cv=10, scoring='f1').mean()
    
    # Armazena as métricas calculadas no dicionário de métricas
    metrics['Acurácia'].append(accuracy)
    metrics['Precisão'].append(precision)
    metrics['Sensibilidade/Recall'].append(recall)
    metrics['Medida-F1'].append(f1)

# Exibir os resultados
for metric, values in metrics.items():
    print(f'{metric}: {values}')
