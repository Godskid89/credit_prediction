import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)


model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory



@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        query = pd.get_dummies(pd.DataFrame(json_))

        query = query.reindex(columns=model_columns, fill_value=0)

        prediction = list(clf.predict(query))

        # Converting to int from int64
        return jsonify({"prediction": list(map(int, prediction))})

    except Exception as e:

        return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    
if __name__ == '__main__':
     clf = joblib.load(model_file_name)
     model_columns = joblib.load(model_columns_file_name)
     app.run(port=8081)