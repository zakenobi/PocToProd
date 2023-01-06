import unittest
from unittest.mock import MagicMock
import tempfile
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from train.train import run
from preprocessing.preprocessing import utils

def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })

class TestPredict(unittest.TestCase):

    # use the function defined on test_predict as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_predict(self):
        # TODO: CODE HERE
        # create a dictionary params for train conf
        params = {
            "batch_size": 1,
            "epochs": 1,
            "dense_dim": 64,
            "min_samples_per_label": 4,
            "verbose": 1
        }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, _ = run.train("fake_path", params, model_dir, False)

            # instance a TextPredictModel class
            textpredictmodel = run.TextPredictionModel.from_artefacts(model_dir)

            # run a prediction
            predictions_obtained = textpredictmodel.predict(['php'], 2)
            print(predictions_obtained)



        # TODO: CODE HERE
        # assert that predictions obtained are equals to expected ones
        # Here we assert that the prediction shape is equal to the one expected
        self.assertGreaterEqual(predictions_obtained.shape, (1, 2))