from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS
from ML_Model.Inference import infer

app= Flask(__name__)
CORS(app)

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


@app.route('/api-predict', methods= ['POST'])
def api_predict():
    print('intruder')
    data= request.get_json(force=True)
    inputs= data['inputData']
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
        output= {'prediction': 'our predictions is: Depressed',
                 'message': "Don't worry about your results, please check out our following suggested articles for your well being:",
                 'article1': '1. https://www.marchofdimes.org/sites/default/files/2023-04/CS_MOD_ISWM_ShortForm_Mindfulness.pdf',
                 'article2': '2. https://www.fammed.wisc.edu/files/webfm-uploads/documents/outreach/im/handout_ppd.pdf',
                 'article3': '3. https://drnozebest.com/blogs/the-doctor-is-in/5-relaxation-techniques-that-support-postpartum-mental-health'}
    elif (prediction[0]==0):
        output= {'prediction': 'our prediction is: Healthy'}
    else:
        output= {'prediction': 'error, pls check your input and try again'}
    print(output)
    return output

if __name__=='__main__':
    app.run(debug=False)