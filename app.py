import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model_rf.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    zero = ["CASH_IN","Cash_In","Cash_in","cash_in","cash_In"]
    one = ["CASH_OUT","Cash_Out", "Cash_out","cash_out","cash_Out"]
    two = ["DEBT","Debt","debt"]
    three = ["PAYMENT","Payment","payment"]
    four = ["TRANSFER","Transfer","transfer"]
    
    int_features = []
    for i in request.form.values():
        if i in one:
            int_features.append(1.0)
        elif i in zero:
            int_features.append(0.0)
        elif i in two:
            int_features.append(2.0)
        elif i in three:
            int_features.append(3.0)
        elif i in four:
            int_features.append(4.0)
        else:
            int_features.append(float(i))
            
    final_features = [np.array(int_features)]

    final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)
    
    output = prediction
    
    if output == [0]:
        output = "Legal Transcation(Transcation is not Fraudualent)"
    elif output == [1]:
        output = "Illegal Transcation(Transcation is Fraudualent)"
    
    return render_template('index.html', prediction_text='Test Result: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
