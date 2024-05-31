import yaml
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from itertools import product

# Load configuration from YAML file
with open("mlflow_test/test_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load data
data = load_iris()
X, y = data.data, data.target

# Define preprocessing steps based on config
steps = []
if config["preprocessing"]["impute_strategy"]:
    steps.append(('imputer', SimpleImputer(strategy=config["preprocessing"]["impute_strategy"])))
if config["preprocessing"]["scale"]:
    steps.append(('scaler', StandardScaler()))

# Add the classifier as the final step
steps.append(('classifier', RandomForestClassifier(random_state=config["model"]["random_state"])))

# Create a pipeline
pipeline = Pipeline(steps)

# Set up the MLflow experiment
mlflow.set_experiment(config["experiment"]["name"])

# Generate all combinations of hyperparameters
param_grid = {
    'classifier__n_estimators': config["model"]["n_estimators"],
    'classifier__max_depth': config["model"]["max_depth"]
}

param_combinations = list(product(param_grid['classifier__n_estimators'], param_grid['classifier__max_depth']))

# Run experiments and log results with MLflow
for n_estimators, max_depth in param_combinations:
    with mlflow.start_run():
        # Update pipeline parameters
        pipeline.set_params(classifier__n_estimators=n_estimators, classifier__max_depth=max_depth)
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("mean_cv_score", cv_scores.mean())
        
        # Log the model
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Logged run with n_estimators={n_estimators}, max_depth={max_depth}, mean_cv_score={cv_scores.mean()}")
