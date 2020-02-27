import os
import pandas as pd
import numpy as np

def prediction_to_csv(prediction: np.ndarray, output_filename: str):
    challenge_dir = os.path.abspath(os.path.dirname(__file__))
    orig_test_data = pd.read_csv(os.path.join(challenge_dir, "data/test.csv"))
    prediction_df = pd.DataFrame.from_dict({"Survived": prediction.tolist()})
    combined_df = pd.concat((orig_test_data["PassengerId"], prediction_df), axis=1)
    if not os.path.exists(os.path.join(challenge_dir, "predictions")):
        os.makedirs(os.path.join(challenge_dir, "predictions"))
    combined_df.to_csv(os.path.join(challenge_dir, "predictions", output_filename), index=False)


if __name__ == "__main__":
    prediction_to_csv(np.array([1,2,3]), "test.csv")
