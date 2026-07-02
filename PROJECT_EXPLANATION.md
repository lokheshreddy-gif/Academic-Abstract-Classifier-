# Transformer-Based Classification for Academic Paper Abstracts

## 📋 Project Overview

This project is an **advanced machine learning system** that automatically classifies academic research paper abstracts into predefined fields of study using state-of-the-art transformer models. The system analyzes the textual content of abstracts and predicts the primary research domain, helping researchers, journal editors, and academic institutions organize papers, route submissions to appropriate reviewers, and maintain structured publication databases.

### Problem Statement

Academic journals and research institutions receive thousands of paper submissions monthly across various disciplines. Manual classification of these papers is:
- **Time-consuming**: Requires domain experts to read and categorize each paper
- **Error-prone**: Human classification can be inconsistent
- **Scalability issue**: Difficult to handle large volumes efficiently

This automated solution addresses these challenges by providing instant, accurate field classification based on abstract text analysis.

---

## 🛠️ Technologies Used

### Backend Technologies

1. **Python 3.10/3.11**
   - Primary programming language
   - Used for data processing, model training, and API development

2. **PyTorch** (`torch>=2.2`)
   - Deep learning framework
   - Used for model training and inference
   - Handles tensor operations and GPU acceleration

3. **Hugging Face Transformers** (`transformers>=4.44.0`)
   - Pre-trained transformer models library
   - Provides DistilBERT model and tokenization utilities
   - Simplifies model loading and fine-tuning

4. **FastAPI** (`fastapi>=0.111.0`)
   - Modern, fast web framework for building APIs
   - Automatic API documentation (Swagger UI)
   - Type validation with Pydantic
   - High performance and async support

5. **Uvicorn** (`uvicorn>=0.30.0`)
   - ASGI server for running FastAPI applications
   - Handles HTTP requests and WebSocket connections
   - Production-ready server implementation

### Data Processing & Analysis

6. **Pandas** (`pandas>=2.2.0`)
   - Data manipulation and analysis
   - CSV file reading and processing
   - Data cleaning and preprocessing

7. **NumPy** (`numpy>=1.26.0`)
   - Numerical computations
   - Array operations for model inputs/outputs
   - Statistical calculations

8. **scikit-learn** (`scikit-learn>=1.4.0`)
   - Machine learning utilities
   - Train/validation/test splitting
   - Evaluation metrics (accuracy, precision, recall, F1-score)
   - Confusion matrix generation

### Visualization

9. **Matplotlib** & **Seaborn**
   - Data visualization
   - Exploratory data analysis plots
   - Statistical visualizations

### Frontend Technologies

10. **HTML5**
    - Structure of the web interface
    - Semantic markup

11. **CSS3**
    - Modern styling with gradients
    - Animations and transitions
    - Responsive design

12. **JavaScript (Vanilla)**
    - Client-side interactivity
    - API calls to FastAPI backend
    - Dynamic content updates

---

## 🤖 Models Used

### Primary Model: DistilBERT

**DistilBERT** (`distilbert-base-uncased`) is the core model used for classification.

#### What is DistilBERT?

- **Distilled version of BERT**: Smaller, faster, and lighter than BERT while retaining 97% of its performance
- **Transformer architecture**: Based on the attention mechanism
- **Pre-trained on large text corpus**: Trained on Wikipedia and BookCorpus
- **Efficient**: 60% faster and 40% smaller than BERT

#### Model Architecture

```
Input Text (Abstract)
    ↓
Tokenization (max_length=128)
    ↓
DistilBERT Encoder (6 layers, 768 hidden dimensions)
    ↓
Classification Head (3 output classes)
    ↓
Softmax Activation
    ↓
Output: Label + Confidence Scores
```

#### Model Configuration

- **Base Model**: `distilbert-base-uncased`
- **Number of Labels**: 3 (cs.CV, cs.LG, stat.ML)
- **Max Sequence Length**: 128 tokens
- **Training**: Fine-tuned on academic abstract dataset
- **Epochs**: 1 (for fast training)
- **Learning Rate**: 2e-5
- **Batch Size**: 16

#### Fine-Tuning Process

1. **Load Pre-trained Model**: Start with DistilBERT pre-trained weights
2. **Add Classification Head**: Custom output layer for 3-class classification
3. **Train on Academic Abstracts**: Fine-tune on ArXiv dataset
4. **Save Model**: Store trained weights and tokenizer for inference

---

## ⚙️ Functionalities

### 1. Data Processing & Preprocessing

- **CSV Data Loading**: Reads academic paper datasets (ArXiv format)
- **Column Auto-Detection**: Automatically identifies abstract and label columns
- **Label Normalization**: Extracts primary label from multi-label formats (e.g., `['cs.CV', 'cs.LG']` → `cs.CV`)
- **Data Cleaning**: Removes missing values, duplicates, and very short abstracts (< 40 characters)
- **Balanced Subset Creation**: Creates balanced training dataset across all classes

### 2. Model Training

- **Pre-trained Model Loading**: Downloads and loads DistilBERT from Hugging Face
- **Custom Dataset Class**: Handles text tokenization and label encoding
- **Stratified Data Splitting**: 80% train, 10% validation, 10% test (stratified by label)
- **Training with Validation**: Monitors performance during training
- **Model Checkpointing**: Saves best model based on F1-score
- **Evaluation Metrics**: Calculates accuracy, precision, recall, F1-score

### 3. Model Evaluation

- **Test Set Evaluation**: Comprehensive evaluation on held-out test set
- **Confusion Matrix**: Detailed per-class performance analysis
- **Misclassified Samples**: Identifies and saves incorrectly predicted examples
- **Per-Class Metrics**: Precision, recall, F1 for each label
- **Macro-Averaged Metrics**: Overall performance across all classes

### 4. Web Application Interface

#### Frontend Features

- **Modern UI Design**: 
  - Gradient backgrounds (cyan → blue → pink → orange)
  - Smooth animations and transitions
  - Responsive layout for different screen sizes
  - Professional medical/academic aesthetic

- **User Input**:
  - Large text area for abstract entry
  - Real-time input validation
  - Keyboard shortcut support (Ctrl+Enter to classify)

- **Results Display**:
  - Prominent prediction label with confidence score
  - Complete score breakdown table
  - Color-coded results for quick assessment
  - Loading states with animated indicators

#### Backend API Endpoints

1. **GET `/health`**
   - Purpose: Health check endpoint
   - Response: `{"status": "ok"}`
   - Use case: Verify server is running

2. **GET `/labels`**
   - Purpose: List all available classification labels
   - Response: `["cs.CV", "cs.LG", "stat.ML"]`
   - Use case: Display available categories in UI

3. **POST `/predict`**
   - Purpose: Classify an academic abstract
   - Request Body: `{"abstract": "text here..."}`
   - Response:
     ```json
     {
       "label": "Computer Vision / AI",
       "score": 0.9234,
       "all_scores": [
         {"label": "Computer Vision / AI", "score": 0.9234},
         {"label": "Machine Learning / AI", "score": 0.0456},
         {"label": "Statistics / Machine Learning", "score": 0.0310}
       ]
     }
     ```
   - Use case: Main classification functionality

4. **GET `/docs`** (Auto-generated)
   - Purpose: Interactive API documentation
   - Swagger UI interface for testing endpoints

### 5. Model Deployment

- **Model Persistence**: Saves trained model, tokenizer, and configuration
- **FastAPI Integration**: Loads model at startup for fast inference
- **GPU/CPU Support**: Automatically uses GPU if available, falls back to CPU
- **Docker Support**: Containerized deployment option
- **Production Ready**: Can be deployed on cloud platforms (Render, Railway, AWS, etc.)

---

## 🔄 Complete Workflow

### Phase 1: Data Collection & Preparation

```
1. Collect Dataset
   └─> ArXiv CSV file with abstracts and categories
   
2. Load & Inspect Data
   └─> Read CSV using pandas
   └─> Display basic statistics
   
3. Data Cleaning
   └─> Remove missing values
   └─> Normalize labels (extract primary category)
   └─> Filter short abstracts
   └─> Remove duplicates
   
4. Feature Extraction
   └─> Abstract length
   └─> Word count
   └─> Sentence count
   └─> Average word length
   
5. Create Balanced Subset
   └─> Sample equal number from each class
   └─> Target: ~800-1200 samples (or full dataset if smaller)
```

### Phase 2: Exploratory Data Analysis (EDA)

```
1. Outlier Detection
   └─> Box plots for all numerical features
   └─> Identify and analyze outliers
   
2. Distribution Analysis
   └─> Histograms with KDE curves
   └─> Understand data spread
   
3. Univariate Analysis
   └─> Individual feature distributions
   └─> Label frequency analysis
   
4. Bivariate Analysis
   └─> Feature relationships (scatter plots)
   └─> Label comparisons (violin plots)
   
5. Correlation Analysis
   └─> Correlation matrix heatmap
   └─> Identify feature dependencies
   
6. Pair Plot Analysis
   └─> Comprehensive pairwise relationships
   └─> Multi-dimensional data visualization
```

### Phase 3: Model Training

```
1. Data Splitting
   └─> Train: 80% (stratified)
   └─> Validation: 10% (stratified)
   └─> Test: 10% (stratified)
   
2. Load Pre-trained Model
   └─> Download DistilBERT from Hugging Face
   └─> Initialize tokenizer
   └─> Configure for 3-class classification
   
3. Tokenization
   └─> Convert text to token IDs
   └─> Truncate/pad to max_length=128
   └─> Create attention masks
   
4. Training Configuration
   └─> Learning rate: 2e-5
   └─> Batch size: 16
   └─> Epochs: 1
   └─> Evaluation strategy: per epoch
   └─> Best model selection: by F1-score
   
5. Model Training
   └─> Fine-tune DistilBERT on training set
   └─> Validate on validation set
   └─> Save best checkpoint
   
6. Model Evaluation
   └─> Predict on test set
   └─> Calculate metrics (accuracy, precision, recall, F1)
   └─> Generate confusion matrix
   └─> Identify misclassified samples
```

### Phase 4: Model Deployment

```
1. Save Model Artifacts
   └─> Model weights (model.safetensors)
   └─> Tokenizer files (tokenizer.json, vocab.txt)
   └─> Configuration (config.json)
   └─> Label mapping (labels.json)
   
2. Create FastAPI Application
   └─> Load saved model and tokenizer
   └─> Define API endpoints
   └─> Implement prediction logic
   
3. Build Web Interface
   └─> HTML structure
   └─> CSS styling
   └─> JavaScript for API calls
   
4. Start Server
   └─> Run: uvicorn app:app --host 0.0.0.0 --port 8001
   └─> Server starts and loads model
```

### Phase 5: User Interaction (Runtime)

```
User Input Flow:
   1. User opens browser → http://localhost:8001/
   2. User pastes abstract text
   3. User clicks "Classify Abstract"
   
Backend Processing:
   4. JavaScript sends POST request to /predict
   5. FastAPI receives request
   6. Tokenize abstract text (max_length=128)
   7. Run model inference (forward pass)
   8. Apply softmax to get probabilities
   9. Map label IDs to friendly names
   10. Sort scores (highest to lowest)
   11. Return JSON response
   
Frontend Display:
   12. JavaScript receives response
   13. Display predicted label and score
   14. Render all scores table
   15. Apply visual styling and animations
```

---

## 📊 Model Performance

### Evaluation Metrics

The fine-tuned DistilBERT model typically achieves:

- **Accuracy**: 0.75 - 0.90 (depending on dataset balance)
- **Macro Precision**: 0.70 - 0.85
- **Macro Recall**: 0.70 - 0.85
- **Macro F1-Score**: 0.70 - 0.85

### Classification Categories

The model classifies abstracts into three domains:

1. **Computer Vision / AI** (cs.CV)
   - Image processing, object detection, computer vision applications

2. **Machine Learning / AI** (cs.LG)
   - Neural networks, deep learning, reinforcement learning, ML algorithms

3. **Statistics / Machine Learning** (stat.ML)
   - Statistical methods, Bayesian models, probabilistic ML, causal inference

---

## 🚀 How to Run

### Prerequisites

1. Install Python 3.10 or 3.11
2. Install required packages:
   ```bash
   pip install torch transformers fastapi uvicorn pandas numpy scikit-learn
   ```

### Training the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train the DistilBERT model
- Evaluate on test set
- Save model to `final_model/` directory
- Generate evaluation metrics and visualizations

### Running the Web Application

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8001
```

Then open your browser and navigate to:
```
http://localhost:8001/
```

### Testing the API

**Health Check:**
```bash
curl http://localhost:8001/health
```

**Get Labels:**
```bash
curl http://localhost:8001/labels
```

**Predict:**
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{"abstract":"Your abstract text here..."}'
```

---

## 📁 Project Structure

```
Gen_Rohit/
├── app.py                          # FastAPI application
├── train_model.py                  # Model training script
├── generate_visualizations.py       # EDA visualization script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── PROJECT_EXPLANATION.md          # This file
├── Dockerfile                      # Docker deployment config
├── project_report.tex              # LaTeX project report
│
├── final_model/                    # Trained model directory
│   ├── config.json                 # Model configuration
│   ├── model.safetensors           # Model weights
│   ├── tokenizer.json              # Tokenizer
│   ├── tokenizer_config.json       # Tokenizer config
│   ├── vocab.txt                   # Vocabulary
│   └── labels.json                 # Label mapping
│
├── train.csv                       # Training dataset
├── val.csv                         # Validation dataset
├── test.csv                        # Test dataset
│
├── metrics.json                    # Evaluation metrics
├── confusion_matrix.csv            # Confusion matrix
├── misclassified_samples.csv       # Misclassified examples
│
└── Visualization Images/
    ├── 1_outlier_analysis.png
    ├── 2_histogram_analysis.png
    ├── 3_univariate_analysis.png
    ├── 4_bivariate_analysis.png
    ├── 5_correlation_analysis.png
    └── 6_pair_plot_analysis.png
```

---

## 🎯 Key Features

1. **Automatic Classification**: Instantly categorizes academic abstracts into research domains
2. **High Accuracy**: Leverages state-of-the-art transformer models
3. **User-Friendly Interface**: Modern, responsive web application
4. **RESTful API**: Easy integration with other systems
5. **Comprehensive Evaluation**: Detailed metrics and analysis
6. **Production Ready**: Can be deployed on cloud platforms
7. **Extensible**: Easy to add more categories or improve model

---

## 🔮 Future Enhancements

- **Multi-Label Classification**: Support papers with multiple research domains
- **Expanded Categories**: Add more academic fields (Physics, Biology, Economics, etc.)
- **Batch Processing**: Handle multiple abstracts simultaneously
- **Model Ensemble**: Combine multiple transformer models for improved accuracy
- **Continuous Learning**: Adapt to new research trends over time
- **API Integration**: Connect with journal management systems
- **Mobile App**: Native mobile application for on-the-go classification

---

## 📚 References

- **DistilBERT Paper**: Sanh et al., "DistilBERT, a distilled version of BERT"
- **Hugging Face Transformers**: https://huggingface.co/transformers/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **ArXiv Dataset**: https://arxiv.org/

---

## 👥 Use Cases

1. **Journal Editorial Management**: Route submissions to appropriate reviewers
2. **Academic Database Organization**: Tag papers with research domains
3. **Conference Paper Routing**: Match papers with expert reviewers
4. **Research Discovery**: Help researchers find papers in their field
5. **Digital Library Management**: Organize large paper collections

---

## ✅ Summary

This project successfully combines:
- **Advanced AI/ML**: State-of-the-art transformer models (DistilBERT)
- **Modern Web Development**: FastAPI backend with responsive frontend
- **Data Science Best Practices**: Comprehensive EDA, proper train/test splits, detailed evaluation
- **Production Deployment**: Docker support, cloud-ready architecture

The result is a **complete, end-to-end machine learning system** that can automatically classify academic paper abstracts with high accuracy, providing a valuable tool for researchers, publishers, and academic institutions.








