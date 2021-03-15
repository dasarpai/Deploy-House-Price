#When a webpage hosted on our website is requested by the some customer's browser then how
#this the server will accept parameters from the webpage, perform the prediction and respond to the service
#That implementation is done here.

import numpy as np
import json


from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_houseprice.pkl', 'rb'))

with open("config.json", "r") as c:
    params = json.load(c)["params"]

print (params)

@app.route('/')
def home():
    return render_template('index.html', params=params)


@app.route("/house_price", methods=['GET', 'POST'])
def house_price():
    return render_template('house_price.html', params=params)

@app.route('/predict',methods=['POST','GET'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    return render_template('house_price.html',params=params, prediction_text= "House Price Shall be Around $: {:,}".format(output) )


if __name__ == "__main__":
    app.run(debug=True, port=5001)