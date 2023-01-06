# Path: predict/predict/app.py
import os
import json
import logging
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from flask import Flask, request

import run

app = Flask(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a route to predict
@app.route("/predict", methods=["POST"])
def predict():
    # get the data from the request
    data = request.get_json(force=True)
    # create a dataframe from the data
    data_df = pd.DataFrame(data)
    # get the model path from the data
    model_path = data_df["model_path"][0]
    # load the model
    model = run.TextPredictionModel.from_artefacts(model_path)
    # predict the data
    predictions = model.predict(data_df["text"])
    # create a response
    response = {"predictions": predictions}
    # return the response
    return json.dumps(response)

# Curl example:
# curl -X POST -H "Content-Type: application/json" -d '{"model_path": "artefacts", "text": ["How to create a new column in pandas dataframe?", "How to convert a pandas dataframe to a numpy array?"]}' http://localhost:5000/predict

# create a route to health check
@app.route("/health", methods=["GET"])
def health():
    return "ok"

# create a hello world route
@app.route("/", methods=["GET"])
def hello():
    return "Hello World!"

# Create a page to test the model with any text
@app.route("/test", methods=["GET"])
def test():
    return """
    <html>
        <body>
            <form action="/predict" method="post">
                <label for="text">Text:</label><br>
                <input type="text" id="text" name="text" value="How to create a new column in pandas dataframe?"><br>
                <input type="hidden" id="model_path" name="model_path" value="artefacts">
                <input type="submit" value="Submit">
            </form>
        </body>
    </html>
    """