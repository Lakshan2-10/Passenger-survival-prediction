import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset into DataFrame
iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = df['target'].apply(lambda x: iris.target_names[x])


# Load model
with open('C:\Users\AKLakshan\Desktop\your project\model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title('Iris Species Classifier')

# Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:', 
                        ['Data Exploration', 'Visualization', 'Prediction'])

# Data Exploration Section
if options == 'Data Exploration':
    st.header('Iris Dataset Exploration')
    st.write("Shape of dataset:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("First 5 rows:", df.head())

# Visualization Section
elif options == 'Visualization':
    st.header('Data Visualizations')
    
    # Pairplot
    st.subheader('Pairplot of Features')
    fig = sns.pairplot(df, hue='species')
    st.pyplot(fig)
    
    # Boxplot
    st.subheader('Boxplot of Features')
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df.drop(columns=['target', 'species']))
    st.pyplot()

# Prediction Section
else:
    st.header('Predict Iris Species')
    
    # Input widgets
    sepal_length = st.slider('Sepal Length', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width = st.slider('Sepal Width', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
    petal_length = st.slider('Petal Length', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width = st.slider('Petal Width', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))
    
    # Make prediction
    if st.button('Predict'):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        st.success(f'The predicted species is: {iris.target_names[prediction[0]]}')
        st.write('Prediction probabilities:')
        for i, prob in enumerate(prediction_proba[0]):
            st.write(f"{iris.target_names[i]}: {prob:.2f}")
