# --- 1. IMPORT LIBRARIES ---
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# --- 2. LOAD THE DATASET ---
iris = load_iris()

# --- 3. PREPARE THE DATA ---
X = iris.data
y = iris.target

# --- 4. CREATE AND TRAIN THE MODEL ---
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# --- 5. CREATE THE WEB APP INTERFACE ---
st.title("Aplikasi Prediksi Spesies Bunga Iris")
st.write("Aplikasi ini memprediksi spesies bunga Iris (Setosa, Versicolor, atau Virginica) berdasarkan pengukuran sepal dan petal.")

# --- User Input Sliders in a sidebar ---
st.sidebar.header("Masukkan Pengukuran Bunga:")

def user_input_features():
    sepal_length = st.sidebar.slider("Panjang Sepal (cm)", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Lebar Sepal (cm)", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Panjang Petal (cm)", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Lebar Petal (cm)", 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# --- Display User Input ---
st.subheader("Pengukuran yang Anda masukkan:")
st.write(df)

# --- Make Prediction ---
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# --- Display Prediction ---
st.subheader("Hasil Prediksi Spesies:")
st.write(iris.target_names[prediction][0].capitalize())

st.subheader("Probabilitas Hasil Prediksi:")
st.write(prediction_proba)