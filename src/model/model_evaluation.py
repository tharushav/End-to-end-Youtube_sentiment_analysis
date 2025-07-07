import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
import mlflow.lightgbm  # Add this import
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> CountVectorizer:
    """Load the saved Bag of Words vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('Bag of Words vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_tracking_uri("http://ec2-13-49-46-71.eu-north-1.compute.amazonaws.com:5000/")

    mlflow.set_experiment('dvc-pipeline-runs')
    
    with mlflow.start_run() as run:
        try:
            # Load parameters from YAML file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters
            for key, value in params.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        mlflow.log_param(f"{key}_{sub_key}", sub_value)
                else:
                    mlflow.log_param(key, value)
            
            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'bow_vectorizer.pkl'))

            # Load test data for signature inference
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Prepare test data
            X_test_bow = vectorizer.transform(test_data['Comment'].values)
            
            # Convert to float32 for LightGBM compatibility
            X_test_bow = X_test_bow.astype(np.float32)
            
            y_test = test_data['Sentiment'].values

            # Create input example with proper feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Create a sample input for signature inference (using dense format)
            sample_input = X_test_bow[:1].toarray()
            input_example = pd.DataFrame(sample_input, columns=feature_names)

            # Use a simple artifact_path name
            model_artifact_path = "model"
            
            # Log model with proper signature and input example
            logger.info(f"Logging model with artifact_path: {model_artifact_path}")
            
            # Use lightgbm.log_model for better compatibility
            try:
                # Create signature with the actual model prediction
                signature = infer_signature(input_example, model.predict(sample_input))
                
                mlflow.lightgbm.log_model(
                    lgb_model=model,
                    artifact_path=model_artifact_path,
                    signature=signature,
                    input_example=input_example
                )
                logger.info("Model logged successfully with lightgbm format")
                
                # Force MLflow to flush artifacts
                mlflow.end_run()
                mlflow.start_run(run_id=run.info.run_id)
                
            except Exception as lightgbm_error:
                logger.warning(f"lightgbm format failed: {lightgbm_error}, trying sklearn format")
                # Fallback to sklearn format with pickle serialization
                try:
                    signature = infer_signature(input_example, model.predict(sample_input))
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=model_artifact_path,
                        signature=signature,
                        input_example=input_example,
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
                    )
                    logger.info("Model logged successfully with sklearn format")
                except Exception as sklearn_error:
                    logger.error(f"Both logging methods failed: lightgbm: {lightgbm_error}, sklearn: {sklearn_error}")
                    # Log as a simple pickle file
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                        pickle.dump(model, tmp_file)
                        tmp_file.flush()
                        mlflow.log_artifact(tmp_file.name, model_artifact_path)
                    logger.info("Model logged as pickle artifact")

            # Wait a moment for the artifact to be properly saved
            import time
            time.sleep(5)

            # Verify the model was logged by listing artifacts
            client = mlflow.tracking.MlflowClient()
            final_model_path = model_artifact_path
            try:
                artifacts = client.list_artifacts(run.info.run_id)
                artifact_paths = [art.path for art in artifacts]
                logger.info(f"Logged artifacts: {artifact_paths}")
                
                # Check if the model artifact exists
                model_found = any(model_artifact_path in path for path in artifact_paths)
                if model_found:
                    logger.info(f"Model artifact confirmed: {model_artifact_path}")
                    final_model_path = model_artifact_path
                else:
                    # Look for any model-related artifacts
                    model_artifacts = [path for path in artifact_paths if 'model' in path.lower() or path.endswith('.pkl')]
                    if model_artifacts:
                        final_model_path = model_artifacts[0]
                        logger.info(f"Found alternative model artifact: {final_model_path}")
                    else:
                        logger.error("No model artifacts found after logging")
                        # Create a simple artifact as fallback
                        model_path = os.path.join(root_dir, 'lgbm_model.pkl')
                        mlflow.log_artifact(model_path, "fallback_model")
                        final_model_path = "fallback_model/lgbm_model.pkl"
                        logger.info(f"Logged fallback model artifact: {final_model_path}")
                        
            except Exception as e:
                logger.warning(f"Could not verify artifacts: {e}")
                final_model_path = model_artifact_path

            # Save model info with the confirmed artifact path
            save_model_info(run.info.run_id, final_model_path, 'experiment_info.json')

            # Log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, 'bow_vectorizer.pkl'))

            # Evaluate model and get metrics (using DataFrame to avoid feature name warnings)
            X_test_df = pd.DataFrame(X_test_bow.toarray(), columns=feature_names)
            report, cm = evaluate_model(model, X_test_df, y_test)

            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, "Test_Data")

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("vectorizer_type", "Bag of Words")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")
            mlflow.set_tag("pipeline_stage", "model_evaluation")
            
            logger.info(f"Model evaluation completed. Run ID: {run.info.run_id}")
            print(f"Model evaluation completed. Run ID: {run.info.run_id}")
            print(f"Model registered with artifact path: {final_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")
            raise

if __name__ == '__main__':
    main()