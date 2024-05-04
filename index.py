from flask import Flask, render_template, request
import pandas as pd
from ML_Model.Inference import infer

app= Flask(__name__)
@app.route('/')
def home():
    print("working")
    return render_template('index.html')

@app.route('/predict', methods= ['POST'])
def predict():
    inputs= [x for x in request.form.values()]
    inputs[0]= float(inputs[0])
    input= pd.DataFrame({'Age':[inputs[0]], 
        'Feeling sad or Tearful':[inputs[1]], 
        'Irritable towards baby & partner':[inputs[2]],
        'Trouble sleeping at night':[inputs[3]],
        'Problems concentrating or making decision':[inputs[4]],
        'Overeating or loss of appetite':[inputs[5]], 
        'Feeling anxious':[inputs[6]], 
        'Feeling of guilt':[inputs[7]],
        'Problems of bonding with baby':[inputs[8]], 
        'Suicide attempt':[inputs[9]]})
    
    prediction= infer(input)
    
    if (prediction[0]==1):
        output= 'Depressed'
    elif (prediction[0]==0):
        output= 'Healthy'
    else:
        output= ''
    
    return render_template('index.html', prediction_text1= output)


if __name__=='__main__':
    app.run(debug=False)