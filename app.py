import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Load the scaler
fp = open("MinMaxscaler.bin", "rb")
scaler = pickle.load(fp)
fp.close()

# Load LogisticRegression
fp = open("RFmodel.bin", "rb")
model2 = pickle.load(fp)
fp.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [[float(x) for x in request.form.values()]]
    final_features = scaler.transform(int_features)
    #output = model.predict(final_features)
    # Now convert linear svm's prediction to probability
    #out1 = sigmoid(model1.decision_function(final_features))
    # Get prediction probability from logistic regression
    final_output= model2.predict_proba(final_features)[:,1]

    #output = prediction.reshape(-1, 1)
    output=(str(round(final_output[0], 2)))
    return render_template('index.html', prediction_text='Probability of Mortality in COVID-19 patient {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)