import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

# Carregando o modelo
modelo = xgb.XGBRegressor()
modelo.load_model("regressor.json")

# Títulos
st.write("Calculadora de imóveis")
st.sidebar.title("Entre com os as características do apartamento")

# Capturando as informações
area = st.sidebar.number_input("Área", min_value=10, max_value=1000, step=25, value=70)
bathrooms = st.sidebar.slider("Banheiros", min_value=1, max_value=5)
garage = st.sidebar.slider("Garagens", min_value=0, max_value=5)
rooms = st.sidebar.slider("Quartos", min_value=1, max_value=5)
condo = st.sidebar.number_input("Valor do condomínio", min_value=10, max_value=3000, step=50, value=70)
estado = st.sidebar.selectbox("Estado", options=[
    'SE', 'AL', 'AM', 'BA', 'CE', 'PB', 'PE', 'RN', 'ES', 'MG', 'PR',
    'RJ', 'RS', 'SP', 'SC', 'DF', 'GO'])

# Criando dataset para aplicação do modelo
apto = pd.DataFrame([[area, rooms, bathrooms, garage, condo, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    columns=["area", "rooms", "bathrooms", "garage", "condo", 'SE', 'AL', 'AM', 'BA', 'CE',
                             'PB', 'PE', 'RN', 'ES', 'MG', 'PR',
                             'RJ', 'RS', 'SP', 'SC', 'DF', 'GO'])

# Modificando valores do dataset com base no estado do imóvel
for col in apto.columns:
    if estado == col:
        apto[col] = 1
    else:
        continue

# Separando as features em grupos
numerical_features = ["area", "rooms", "bathrooms", "garage", 'condo']
categorical_features = ['SE', 'AL', 'AM', 'BA', 'CE','PB', 'PE', 'RN', 'ES', 'MG', 'PR',
                        'RJ', 'RS', 'SP', 'SC', 'DF', 'GO']

# Normalização dos dados
scaler = MinMaxScaler()
data_pipeline = ColumnTransformer([("numerical", scaler, numerical_features)],
                                  remainder="passthrough")

# Aplicação do modelo
x = apto[numerical_features + categorical_features]
y = data_pipeline.fit_transform(x)
prediction = round(modelo.predict(apto)[0], 2)

# Apresentando o resultado
st.write(f"Preço previsto: R$ {prediction}")
