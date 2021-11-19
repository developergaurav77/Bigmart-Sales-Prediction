import re
from flask import Flask,render_template,request,redirect,url_for
import numpy as np
import joblib

model_path = 'models/model.sav'

app = Flask(__name__)

@app.route('/<result>')
def predict(result):
    return f"<h1>prediction is: {result}</h1>"

@app.route('/',methods=["POST","GET"])
def prediction():

    if request.method == 'POST':
        item_weight = float(request.form['item_weight'])
        item_fat_content = float(request.form['item_fat_content'])
        item_visibility = float(request.form['item_visibility'])
        item_type = float(request.form['item_type'])
        item_mrp = float(request.form['item_mrp'])
        outlet_establishment_year = float(request.form['outlet_establishment_year'])
        outlet_size = float(request.form['outlet_size'])
        outlet_location_type = float(request.form['outlet_location_type'])
        outlet_type = float(request.form['outlet_type'])

        content = np.array([[item_weight,item_fat_content,item_visibility,item_type,item_mrp,outlet_establishment_year,outlet_size,outlet_location_type,outlet_type]])

        model = joblib.load(model_path)
        predictionn = model.predict(content)
        print(predictionn)

        # return redirect(url_for("predict",result = predictionn))
        
        return redirect(url_for("predict",result=predictionn))


    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run()