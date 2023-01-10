# Home Credit Dataset EDA and repayment prediction

This is my capstone project from Turing College, based on Home Credit Kaggle Dataset. The goal of this project is to explore this data and build some ML models, which can be used for risk evaluation as service. Here is a brief desciption of the files:

- `EDA.ipynb` - Task formulation, my workflow and Exploratory Data Analysis of the dataset
  - My own packaged `EDAwesome` is used for this task
  - Custom visualizations are created using `seaborn` and `matplotlib`
- `ML_downsampled.ipynb` - Machine Learning models
  - Bayesian hyperparameter tuning with Optuna
  - Simple NLP with SpaCy, tokens clusterization techniques
- `status_prediction_api.py` - FastAPI app for status prediction
- `data_preparation.py` - data preparation for the app
- `models` directory - contains trained models

There are som other notebooks with different ML experiments.