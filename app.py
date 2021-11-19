from flask import Flask, jsonify, render_template, request,url_for
import joblib
import os
import numpy as np
from werkzeug.utils import redirect


app = Flask(__name__)


scalerr = joblib.load('scaler.pkl')
model = joblib.load('rf_model.sav')

@app.route("/<result>")
def predict(result):
    return f"Sales Prediction is: {result}"

@app.route('/',methods=['POST','GET'])
def result():

    if request.method == 'POST':

    

        item_weight= float(request.form['item_weight'])
        item_fat_content=float(request.form['item_fat_content'])
        item_visibility= float(request.form['item_visibility'])
        item_type= float(request.form['item_type'])
        item_mrp = float(request.form['item_mrp'])
        outlet_establishment_year= float(request.form['outlet_establishment_year'])
        outlet_size= float(request.form['outlet_size'])
        outlet_location_type= float(request.form['outlet_location_type'])
        outlet_type= float(request.form['outlet_type'])

        X= np.array([[ item_weight,item_fat_content,item_visibility,item_type,item_mrp,
                  outlet_establishment_year,outlet_size,outlet_location_type,outlet_type ]])



        pred=model.predict(X)

        return redirect(url_for("predict",result=float(pred)))
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run()
