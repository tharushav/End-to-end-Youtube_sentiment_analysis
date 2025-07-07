# register model

import json
import mlflow
import logging
import os

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-13-49-46-71.eu-north-1.compute.amazonaws.com:5000/")

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        # Verify the model exists before attempting registration
        client = mlflow.tracking.MlflowClient()
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        
        # Check if the artifact exists
        try:
            artifacts = client.list_artifacts(run_id)
            artifact_paths = [art.path for art in artifacts]
            logger.info(f"Available artifacts in run {run_id}: {artifact_paths}")
            
            if model_path not in artifact_paths:
                logger.warning(f"Model path '{model_path}' not found in artifacts. Available paths: {artifact_paths}")
                # Try to find the model artifact automatically
                model_artifacts = [path for path in artifact_paths if 'model' in path.lower()]
                if model_artifacts:
                    model_path = model_artifacts[0]
                    logger.info(f"Using found model artifact: {model_path}")
                else:
                    raise ValueError(f"No model artifacts found in run {run_id}")
                    
        except Exception as e:
            logger.error(f"Error checking artifacts: {e}")
            # Continue with original path as fallback
        
        model_uri = f"runs:/{run_id}/{model_path}"
        logger.info(f"Attempting to register model from URI: {model_uri}")
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Set model version alias instead of using deprecated stages
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )
        
        logger.debug(f'Model {model_name} version {model_version.version} registered with alias "staging".')
        return model_version
        
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        # Automatically load from experiment_info.json created by model_evaluation.py
        model_info_path = 'experiment_info.json'
        
        # Check if the file exists
        if not os.path.exists(model_info_path):
            logger.error(f'experiment_info.json not found. Please run model evaluation first: python src/model/model_evaluation.py')
            print(f"Error: {model_info_path} not found.")
            print("Please run the model evaluation stage first to generate the experiment info:")
            print("  python src/model/model_evaluation.py")
            print("Or run the full pipeline:")
            print("  dvc repro")
            return
        
        # Load model info automatically
        model_info = load_model_info(model_info_path)
        
        # Validate required fields
        if 'run_id' not in model_info or 'model_path' not in model_info:
            logger.error('Invalid experiment_info.json format. Missing run_id or model_path.')
            print("Error: Invalid experiment_info.json format.")
            print("The file should contain 'run_id' and 'model_path' fields.")
            return
        
        model_name = "yt_chrome_plugin_model"
        logger.info(f'Registering model with run_id: {model_info["run_id"]}')
        print(f"Registering model '{model_name}' with run_id: {model_info['run_id']}")
        
        register_model(model_name, model_info)
        
        print(f"Model '{model_name}' successfully registered!")
        
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()