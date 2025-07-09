# YouTube Sentiment Analysis Extension

An end-to-end machine learning project that analyzes YouTube video comments in real-time through a Chrome extension. The system uses LightGBM for sentiment classification, MLflow for model management, DVC for pipeline orchestration, and flask for the backend service, with deployment on AWS infrastructure.

## 🚀 Features

- **Real-time Sentiment Analysis**: Analyze YouTube video comments instantly
- **Chrome Extension**: Easy-to-use browser extension interface
- **Interactive Visualizations**: Pie charts, word clouds, and sentiment trend graphs
- **MLOps Pipeline**: Complete ML pipeline with experiment tracking and model versioning
- **Cloud Deployment**: Scalable deployment on AWS infrastructure
- **Comprehensive Metrics**: Comment analysis summary with detailed insights

## 🛠️ Technologies Used

### Machine Learning & Data Processing
- **LightGBM** - Gradient boosting framework for sentiment classification
- **NLTK** - Natural language processing and text preprocessing
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - Machine learning utilities and model evaluation

### MLOps & Pipeline Management
- **MLflow** - Experiment tracking, model registry
- **DVC** - Data versioning and ML pipeline orchestration
- **Docker** - Containerization for consistent deployment

### Backend & API
- **Flask** - REST API development with CORS support
- **YouTube Data API v3** - Fetching YouTube comments

### Frontend & Extension
- **Chrome Extension API** - Browser extension development
- **JavaScript** - Frontend logic and API integration
- **HTML/CSS** - Modern UI with gradient designs and animations

### Cloud & Infrastructure
- **AWS (EC2, S3, ECR)** - Cloud deployment, Cloud storage, To save docker images

### Data Visualization
- **Matplotlib** - Chart generation and data visualization
- **WordCloud** - Text visualization for comment analysis
- **Seaborn** - Statistical data visualization


### Cloud & Infrastructure
- **AWS (EC2, ELB)** - Cloud deployment
- **Databricks SDK** - Integration with Databricks platform

### Data Visualization
- **Matplotlib** - Chart generation and data visualization
- **WordCloud** - Text visualization for comment analysis
- **Seaborn** - Statistical data visualization

## 📂 Project Structure

```
├── backend/
│   └── main.py                 # Flask backend server
├── frontend/
│   ├── popup.html             # Chrome extension popup interface
│   ├── popup.js               # Frontend JavaScript logic
│   └── manifest.json          # Chrome extension manifest
├── src/
│   ├── data/
│   │   ├── data_ingestion.py      # Data collection and splitting
│   │   └── data_preprocessing.py   # Text preprocessing pipeline
│   └── model/
│       ├── model_building.py      # Model training and optimization
│       ├── model_evaluation.py    # Model evaluation and metrics
│       └── model_register.py      # MLflow model registration
├── data/
│   ├── raw/                   # Raw training data
│   └── interim/               # Processed training data
├── dvc.yaml                   # DVC pipeline configuration
├── params.yaml                # Model and pipeline parameters
├── requirements.txt           # Python dependencies
└── setup.py                   # Package setup configuration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- YouTube Data API key
- AWS account (for cloud deployment)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd End-to-end-Youtube_sentiment_analysis
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install the package**
```bash
pip install -e .
```

4. **Set up environment variables**
```bash
export YOUTUBE_API_KEY="your_youtube_api_key"
export MLFLOW_TRACKING_URI="your_mlflow_server_url"
```

### Running the ML Pipeline

1. **Execute the complete DVC pipeline**
```bash
dvc repro
```

2. **Or run individual stages**
```bash
# Data ingestion
dvc repro data_ingestion

# Data preprocessing
dvc repro data_preprocessing

# Model building
dvc repro model_building

# Model evaluation
dvc repro model_evaluation

# Model registration
dvc repro model_registration
```

### Running the Backend Server

1. **Start the Flaskserver**
```bash
cd backend
python main.py
```

### Installing the Chrome Extension

1. **Open Chrome and navigate to `chrome://extensions/`**
2. **Enable Developer mode**
3. **Click "Load unpacked" and select the `frontend/` directory**
4. **The extension should now appear in your Chrome toolbar**

## 📊 Model Performance

The LightGBM model is configured with the following parameters:
- **Learning Rate**: 0.09
- **Max Depth**: 20
- **N Estimators**: 367
- **Max Features**: 1000
- **N-gram Range**: [1, 1]

## 🔧 Configuration

### Model Parameters (`params.yaml`)
```yaml
data_ingestion:
  test_size: 0.20

model_building:
  ngram_range: [1, 1]  
  max_features: 1000
  learning_rate: 0.09
  max_depth: 20
  n_estimators: 367
```

### API Endpoints
- `GET /` - Health check and model status
- `POST /predict` - Batch sentiment prediction
- `POST /predict_with_timestamps` - Sentiment prediction with timestamps
- `POST /generate_chart` - Generate sentiment distribution chart
- `POST /generate_wordcloud` - Generate word cloud visualization
- `POST /generate_trend_graph` - Generate sentiment trend graph
- `GET /health` - Detailed health check with system status

## 🌐 Deployment

The project supports deployment on AWS infrastructure:
- **EC2 instances** for model serving
- **MLflow server** for model management
- **Docker containers** for consistent deployment

