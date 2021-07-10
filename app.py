# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 14:23:13 2021

@author: HP OMEN
"""

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
import json
import config
import pickle
import io


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# render home page


@app.route('/')
def home():
    return render_template('crop_index.html')

# @app.route('/crop-recommend')
# def crop_recommend():
#     return render_template('crop.html')

# RENDER PREDICTION PAGES

# render crop recommendation result page


@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("kota")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop_result.html', prediction=final_prediction)

        else:

            return render_template('try_again.html')


# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = 'e75ecc3cff545a6827001fb71b8d0e54'
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "q=" + city_name + "&APPID=" + api_key
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None
    
    
if __name__ == "__main__":
    app.run(debug=True)