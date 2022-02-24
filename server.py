 
from flask import Flask, jsonify, make_response, request, abort
import pandas as pd
import catboost
import pickle
import sys
from sklearn import preprocessing 
from flask_cors import CORS,cross_origin
model = pickle.load(open("final_model.sav", "rb"))

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)
@app.errorhandler(404)

def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route("/")
def hello():
    print('Hello world!', file=sys.stderr)
    return "Hello World!"

@app.route("/get_prediction", methods=['POST','OPTIONS'])
@cross_origin()
def get_prediction():
  if not request.json:
    abort(400)
  df = pd.DataFrame(request.json, index=[0])
  cols=["CONSOLE","YEAR","CATEGORY","PUBLISHER","RATING","CRITICS_POINT","USER_POINT"]
  df = df[cols]
  print(df, file=sys.stderr)
  le = preprocessing.LabelEncoder()
  df['CONSOLE'] = le.fit_transform(df['CONSOLE'])
  df['CATEGORY'] = le.fit_transform(df['CATEGORY'])
  df['PUBLISHER'] = le.fit_transform(df['PUBLISHER'])
  df['RATING'] = le.fit_transform(df['RATING'])
  # df['USER_POINT'] = le.fit_transform(df['USER_POINT'])
  # le.fit(df)
  # print(le, file=sys.stderr)
  df = df[cols]
  print(df, file=sys.stderr)
  return jsonify({'result': model.predict(df)[0]}), 201
if __name__ == "__main__":
  app.run()