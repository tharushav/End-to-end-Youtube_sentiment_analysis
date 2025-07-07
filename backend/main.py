import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import os
import logging
import threading
import signal
from contextlib import contextmanager

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        if not comment or not isinstance(comment, str):
            return ""
            
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        try:
            stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
            comment = ' '.join([word for word in comment.split() if word not in stop_words])
        except:
            # If stopwords not available, skip this step
            pass

        # Lemmatize the words
        try:
            lemmatizer = WordNetLemmatizer()
            comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        except:
            # If lemmatizer not available, skip this step
            pass

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return str(comment) if comment else ""

# Load the model and vectorizer with better error handling
@contextmanager
def timeout(seconds):
    """Context manager to timeout operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

def load_model_and_vectorizer_safe():
    """Load model and vectorizer with fallback options - prioritize EC2/MLflow first."""
    
    # First try to load from MLflow (EC2) with proper timeout
    try:
        # Set environment variables for MLflow timeouts
        os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '3'  
        os.environ['MLFLOW_HTTP_REQUEST_MAX_RETRIES'] = '1'
        
        # Use timeout context manager for the entire MLflow operation
        with timeout(5): 
            # Try to load from MLflow model registry
            mlflow.set_tracking_uri("http://ec2-13-49-46-71.eu-north-1.compute.amazonaws.com:5000/")
            client = MlflowClient()
            model_name = "yt_chrome_plugin_model"
            
            # Set a timeout for socket operations
            import socket
            socket.setdefaulttimeout(2)  
            
            try:
                # Get the latest version with staging alias
                model_version_info = client.get_model_version_by_alias(model_name, "staging")
                model_version = model_version_info.version
            except Exception as alias_error:
                # Fallback to latest version
                try:
                    latest_versions = client.get_latest_versions(model_name, stages=["None"])
                    if latest_versions:
                        model_version = latest_versions[0].version
                    else:
                        raise Exception("No model versions found")
                except Exception as latest_error:
                    model_version = "1"  # Default fallback
            
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Load vectorizer from local file (still needed for MLflow model)
            vectorizer_path = "bow_vectorizer.pkl"
            if not os.path.isabs(vectorizer_path):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                vectorizer_path = os.path.join(project_root, vectorizer_path)
            
            with open(vectorizer_path, 'rb') as file:
                vectorizer = pickle.load(file)
                
            logger.info("Model loaded from MLflow") 
            return model, vectorizer, "mlflow"
        
    except (TimeoutError, ConnectionError, OSError, Exception) as mlflow_error:
        logger.warning(f"MLflow/EC2 unavailable, using local files") 
        # Reset socket timeout
        import socket
        socket.setdefaulttimeout(None)
    
    # If MLflow fails, fall back to local files
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "lgbm_model.pkl")
        vectorizer_path = os.path.join(project_root, "bow_vectorizer.pkl")
        
        # Check if local files exist
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            
            with open(vectorizer_path, 'rb') as file:
                vectorizer = pickle.load(file)
                
            logger.info("Model loaded from local files")  
            return model, vectorizer, "local"
        else:
            logger.warning(f"Local model files not found") 
            
    except Exception as local_error:
        logger.warning(f"Failed to load from local files: {local_error}")
        
    # Final fallback attempt for local files with different paths
    try:
        # Try different possible locations for local files
        possible_paths = [
            os.path.dirname(os.path.abspath(__file__)),  # Same directory as main.py
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),  # models subdirectory
            "/Users/tharushavihanga/Developer/End-to-end-Youtube_sentiment_analysis",  # project root
        ]
        
        for base_path in possible_paths:
            model_path = os.path.join(base_path, "lgbm_model.pkl")
            vectorizer_path = os.path.join(base_path, "bow_vectorizer.pkl")
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
                
                with open(vectorizer_path, 'rb') as file:
                    vectorizer = pickle.load(file)
                    
                logger.info(f"Model loaded from fallback location")  # Simplified log message
                return model, vectorizer, "local_fallback"
        
        raise Exception("No local model files found in any expected location")
        
    except Exception as final_error:
        logger.error(f"Final fallback failed: {final_error}")
        raise Exception(f"Could not load model from any source. Check if model files exist in the project directory.")

# Initialize the model and vectorizer
try:
    model, vectorizer, model_source = load_model_and_vectorizer_safe()
    logger.info(f"Model initialized from {model_source}")  # Simplified log message
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    model, vectorizer, model_source = None, None, None

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to our flask api",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_source": model_source
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 500
        
    try:
        data = request.json
        comments = data.get('comments')
        
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Removed verbose logging here

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert to float32 for LightGBM compatibility
        transformed_comments = transformed_comments.astype(np.float32)

        # Make predictions based on model source
        if model_source == "mlflow":
            # For MLflow models, convert sparse matrix to DataFrame with feature names
            feature_names = vectorizer.get_feature_names_out()
            transformed_df = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)
            predictions = model.predict(transformed_df)
        else:
            # For local models, use sparse matrix directly
            predictions = model.predict(transformed_comments)
        
        # Ensure predictions are in the right format - convert to integers first
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        
        # Convert predictions to proper integer format
        processed_predictions = []
        for pred in predictions:
            try:
                # Handle different prediction formats
                if isinstance(pred, str):
                    # Map string labels to numeric values
                    if pred.lower() in ['positive', '1']:
                        processed_predictions.append("1")
                    elif pred.lower() in ['negative', '-1']:
                        processed_predictions.append("-1")
                    elif pred.lower() in ['neutral', '0']:
                        processed_predictions.append("0")
                    else:
                        processed_predictions.append("0")  # Default to neutral
                else:
                    # Handle numeric predictions
                    int_pred = int(float(pred))
                    processed_predictions.append(str(int_pred))
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting prediction {pred}: {e}")
                processed_predictions.append("0")  # Default to neutral
        
        # Removed verbose logging here
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, processed_predictions)]
    return jsonify(response)

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 500
        
    try:
        data = request.json
        comments_data = data.get('comments')
        
        if not comments_data:
            return jsonify({"error": "No comments provided"}), 400

        # Removed verbose logging here

        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert to float32 for LightGBM compatibility
        transformed_comments = transformed_comments.astype(np.float32)
        
        # Make predictions based on model source
        if model_source == "mlflow":
            # For MLflow models, convert sparse matrix to DataFrame with feature names
            feature_names = vectorizer.get_feature_names_out()
            transformed_df = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)
            predictions = model.predict(transformed_df)
        else:
            # For local models, use sparse matrix directly
            predictions = model.predict(transformed_comments)
        
        # Ensure predictions are in the right format
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        
        # Convert predictions to proper integer format
        processed_predictions = []
        for pred in predictions:
            try:
                # Handle different prediction formats
                if isinstance(pred, str):
                    # Map string labels to numeric values
                    if pred.lower() in ['positive', '1']:
                        processed_predictions.append("1")
                    elif pred.lower() in ['negative', '-1']:
                        processed_predictions.append("-1")
                    elif pred.lower() in ['neutral', '0']:
                        processed_predictions.append("0")
                    else:
                        processed_predictions.append("0")  # Default to neutral
                else:
                    # Handle numeric predictions
                    int_pred = int(float(pred))
                    processed_predictions.append(str(int_pred))
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting prediction {pred}: {e}")
                processed_predictions.append("0")  # Default to neutral
        
        # Removed verbose logging here
        
    except Exception as e:
        logger.error(f"Prediction with timestamps failed: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} 
                for comment, sentiment, timestamp in zip(comments, processed_predictions, timestamps)]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for debugging."""
    # Check if EC2 instance is reachable
    ec2_status = "unknown"
    try:
        import socket
        socket.setdefaulttimeout(5)
        mlflow.set_tracking_uri("http://ec2-13-49-46-71.eu-north-1.compute.amazonaws.com:5000/")
        client = MlflowClient()
        client.list_experiments()
        ec2_status = "reachable"
    except Exception as e:
        ec2_status = f"unreachable: {str(e)}"
    
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_source": model_source,
        "ec2_status": ec2_status,
        "nltk_data_available": {
            "stopwords": True,
            "wordnet": True
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)