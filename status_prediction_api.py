import pandas as pd
import numpy as np
import joblib, os

from data_preparation import get_ready_pipeline

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class InputData(BaseModel):

    amt_income_total : float
    amt_credit: float
    days_birth: int
    days_employed: int

    name_education_type: str
    code_gender: str


# Load the model
model = joblib.load(os.path.join('models', 'test_model.joblib'))

# Load preparation pipeline
preparation_pipeline = get_ready_pipeline()

# Initialize the app
app = FastAPI()

# Get example data from training set, first 5 rows
example_data = pd.read_csv(
    os.path.join('data', 'downsampled_data', 'part-00000-2ccbea9d-88f8-498a-b0b9-bc5808ac195f-c000.csv')
).head()

# Define the root route
@app.get('/')
def root():
    return {'message': 'Welcome to the loan status prediction API!'}

@app.post('/predict')
def predict(data: InputData):

    # Convert the input data to a dataframe, using example data to get the column names
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df.reindex(columns=example_data.columns, fill_value=np.nan)

    # Apply the preparation pipeline
    status_df = pd.read_csv(os.path.join('data', 'downsampled_data', 'part-00000-2ccbea9d-88f8-498a-b0b9-bc5808ac195f-c000.csv'))
    df_prepared = preparation_pipeline.transform(status_df)
    predictions = model.predict(df_prepared)

    return 'Repaid' if predictions[0] == 0 else 'Not repaid'

# Show example InputData
@app.get('/example')
def example():
    selected_feautures = InputData.schema()['properties'].keys()
    return InputData(
        **example_data[selected_feautures].iloc[0].to_dict()
    )

if __name__ == '__main__':
    uvicorn.run(app)