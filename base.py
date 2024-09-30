from flask import Flask, render_template, request
import joblib
import numpy as np
import pickle
import pandas as pd



app = Flask(__name__)

with open(file='laptop_price_prediction_xg_model.pkl', mode='rb') as file:
    model = pickle.load(file)



# with open(file='one_hot_encoder.pkl', mode='rb') as file:
#     encoder = pickle.load(file)


print(type(model))

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict_laptop_price():


    company = request.form.get('Company')
    typename = request.form.get('TypeName')
    ram = request.form.get('Ram')
    opsys = request.form.get('OpSys')
    weight = request.form.get('Weight')
    touchscreen = request.form.get('Touchscreen')
    ips = request.form.get('IPS')
    ppi = request.form.get('PPI')
    cpu_name = request.form.get('cpu_name')
    memory = request.form.get('memory')
    ssd = request.form.get('SSD')
    gpu_brand = request.form.get('Gpu_brand')

    input_data = pd.DataFrame({
        'Company': [company],
        'TypeName': [typename],
        'Ram': [ram],
        'OpSys': [opsys],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'IPS': [ips],
        'PPI': [ppi],
        'cpu_name': [cpu_name],
        'memory': [memory],
        'SSD': [ssd],
        'Gpu_brand': [gpu_brand]
    })


    # print(input_data)

    # Make prediction using the model
    try:
        prediction = model.predict(input_data)
        predicted_price = np.exp(prediction)  # Round the result to 2 decimal places
    except Exception as e:
        predicted_price = f"Error in prediction: {e}"


    
    return render_template('index.html',company=company,typename=typename,ram=ram, opsys=opsys,weight=weight,touchscreen=touchscreen, ips=ips, ppi=ppi, cpu_name=cpu_name,memory=memory,ssd=ssd,gpu_brand=gpu_brand, predicted_price=predicted_price[0])




if __name__ == '__main__':
    app.run(debug=True)