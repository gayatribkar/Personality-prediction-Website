from  flask import Flask, request, url_for, redirect, render_template, jsonify
# from flask_ngrok import run_with_ngrok
import pickle
import flask
import numpy as np
app = Flask(__name__)
# run_with_ngrok(app)

@app.route('/')


def predict():
   return flask.render_template('home.html')

def ValuePredictor(to_predict_list):
   pred_list = np.reshape(to_predict_list, (1,7))
   # pred_list = np.reshape(to_predict_list, (1,7))
   # to_predict = np.array(to_predict_list).reshape(3, 7)
   loaded_model = pickle.load(open("personality_model.pkl", "rb"))
   result = loaded_model.predict(pred_list)
   return result[0]

@app.route('/result', methods=['GET', 'POST'])
def result():
   if request.method == 'POST':
       to_predict_list = request.form.getlist('option')
       print('Done')

       # to_predict_list = list(to_predict_list.values())
       # to_predict_list = list(map(int, to_predict_list))
       result = ValuePredictor([to_predict_list])

       # if(result=='responsible'):
       return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=6060, debug=True)
