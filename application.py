from joblib import load
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Cargar el modelo
with open('./model/mejor_modelo.pkl', 'rb') as f:
    modelo = pickle.load(f)

preprocesador = load('./model/preprocessor.joblib')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/formulario')
def formulario():
    return render_template('formulario.html')


@app.route('/resultado', methods=['POST'])
def resultado():
    # Obtener los datos del formulario
    edad = int(request.form['edad'])
    genero = request.form['genero']
    poliuria = int(request.form['poliuria'])
    polidipsia = int(request.form['polidipsia'])
    perdida_subita_de_peso = int(request.form['perdida_subita_de_peso'])
    debilidad = int(request.form['debilidad'])
    polifagia = int(request.form['polifagia'])
    candidiasis_genital = int(request.form['candidiasis_genital'])
    vision_borrosa = int(request.form['vision_borrosa'])
    picazon = int(request.form['picazon'])
    irritabilidad = int(request.form['irritabilidad'])
    curacion_lenta = int(request.form['curacion_lenta'])
    paresia_parcial = int(request.form['paresia_parcial'])
    rigidez_muscular = int(request.form['rigidez_muscular'])
    alopecia = int(request.form['alopecia'])
    obesidad = int(request.form['obesidad'])

    # Definir los nombres de las columnas
    column_names = ['edad', 'genero', 'poliuria', 'polidipsia', 'perdida_subita_de_peso', 'debilidad', 'polifagia',
                    'candidiasis_genital', 'vision_borrosa', 'picazon', 'irritabilidad', 'curacion_lenta', 'paresia_parcial', 'rigidez_muscular', 'alopecia', 'obesidad']

    datos = np.array([[edad, genero, poliuria, polidipsia, perdida_subita_de_peso, debilidad, polifagia,
                     candidiasis_genital, vision_borrosa, picazon, irritabilidad, curacion_lenta, paresia_parcial, rigidez_muscular, alopecia, obesidad]])

    data_pred = pd.DataFrame(datos, columns=column_names)

    # Preprocesar los datos
    data_preprocesada = preprocesador.transform(data_pred)

    # Realizar la predicción
    prediccion = modelo.predict(data_preprocesada)

    # Obtener el resultado
    resultado = ''
    if prediccion == 1:
        resultado = 'Positivo'
    else:
        resultado = 'Negativo'

    # Renderizar el resultado
    return render_template('resultado.html', resultado=resultado)


if __name__ == '__main__':
    app.run(debug=True)
