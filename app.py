from flask import Flask, render_template, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

def predict(values, dic):
    print("inside")
    if len(values) == 22:
        model = pickle.load(open('models/breast_cancer1.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 8:
        dic2 = {'NewBMI_Obesity 1': 0, 'NewBMI_Obesity 2': 0, 'NewBMI_Obesity 3': 0, 'NewBMI_Overweight': 0,
                'NewBMI_Underweight': 0, 'NewInsulinScore_Normal': 0, 'NewGlucose_Low': 0,
                'NewGlucose_Normal': 0, 'NewGlucose_Overweight': 0, 'NewGlucose_Secret': 0}
        try:
            if dic['BMI'] <= 18.5:
                dic2['NewBMI_Underweight'] = 1
            elif 18.5 < dic['BMI'] <= 24.9:
                pass
            elif 24.9 < dic['BMI'] <= 29.9:
                dic2['NewBMI_Overweight'] = 1
            elif 29.9 < dic['BMI'] <= 34.9:
                dic2['NewBMI_Obesity 1'] = 1
            elif 34.9 < dic['BMI'] <= 39.9:
                dic2['NewBMI_Obesity 2'] = 1
            elif dic['BMI'] > 39.9:
                dic2['NewBMI_Obesity 3'] = 1
        except Exception as e:
            print(e)  

        if 16 <= dic['Insulin'] <= 166:
            dic2['NewInsulinScore_Normal'] = 1

        if dic['Glucose'] <= 70:
            dic2['NewGlucose_Low'] = 1
        elif 70 < dic['Glucose'] <= 99:
            dic2['NewGlucose_Normal'] = 1
        elif 99 < dic['Glucose'] <= 126:
            dic2['NewGlucose_Overweight'] = 1
        elif dic['Glucose'] > 126:
            dic2['NewGlucose_Secret'] = 1
        
        dic.update(dic2)
        values2 = list(map(float, list(dic.values())))
      
        try:
            model = pickle.load(open('models/diabetes.pkl','rb'))
            values = np.asarray(values2)
        except Exception as e:
            print(e)
        print(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/about", methods=['GET', 'POST'])
def aboutPage():
    return render_template('about.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            print(to_predict_dict)
        
            for key, value in to_predict_dict.items():
                try:
                    to_predict_dict[key] = int(value)
                except ValueError:
                    to_predict_dict[key] = float(value)

            to_predict_list = list(map(float, list(to_predict_dict.values())))
            print(to_predict_list)
            pred = predict(to_predict_list, to_predict_dict)
            print(pred)
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred = pred)

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message = message)
    return render_template('malaria_predict.html', pred = pred)


if __name__ == '__main__':
	app.run(debug = True)