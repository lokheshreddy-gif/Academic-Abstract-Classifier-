# 📚 Academic Abstract Classifier

Academic Abstract Classifier is a machine learning-based Natural Language Processing (NLP) application that automatically categorizes research paper abstracts into predefined academic domains. The project leverages classical machine learning algorithms combined with text preprocessing and feature engineering to achieve accurate and efficient document classification.

---

## 🚀 Features

- 📄 Automatic classification of academic paper abstracts
- 🧹 Advanced text preprocessing and cleaning
- 🔤 TF-IDF feature extraction for text representation
- 🤖 Multiple machine learning models for comparison
- 📊 Model evaluation using cross-validation
- ⚙️ Hyperparameter tuning for improved performance
- 📈 Performance metrics including Accuracy, Precision, Recall, and F1-Score
- 📉 Confusion Matrix visualization

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Matplotlib
- Seaborn

### Machine Learning Models
- Logistic Regression
- Support Vector Machine (SVM)
- Multinomial Naive Bayes

### Feature Engineering
- TF-IDF Vectorization

---

## 📂 Project Structure

```
Academic-Abstract-Classifier/
│
├── dataset/                 # Dataset files
├── notebooks/               # Jupyter notebooks
├── models/                  # Saved trained models
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## ✨ Key Functionalities

### Text Preprocessing
- Lowercasing
- Tokenization
- Stopword removal
- Punctuation removal
- Stemming/Lemmatization

### Feature Extraction
- TF-IDF Vectorization
- Vocabulary generation
- Sparse feature representation

### Machine Learning Models
- Logistic Regression
- Support Vector Machine (SVM)
- Multinomial Naive Bayes

### Model Evaluation
- Cross-validation
- Hyperparameter tuning using GridSearchCV
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Academic-Abstract-Classifier.git
```

Navigate to the project directory

```bash
cd Academic-Abstract-Classifier
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python src/train.py
```

### Evaluate the model

```bash
python src/evaluate.py
```

### Predict a new abstract

```bash
python src/predict.py
```

---

## 📊 Model Pipeline

1. Load dataset
2. Text preprocessing
3. TF-IDF vectorization
4. Train multiple classifiers
5. Hyperparameter tuning
6. Cross-validation
7. Model evaluation
8. Predict category for new abstracts

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Cross-Validation Score
- Confusion Matrix

---

## 📸 Screenshots

Include screenshots for:

- Dataset overview
- Text preprocessing output
- Model training
- Accuracy comparison
- Confusion matrix
- Prediction results

---

## 🎯 Future Enhancements

- Deep Learning models (LSTM, GRU)
- Transformer-based models (BERT, RoBERTa)
- Multi-label document classification
- Web-based prediction interface
- Explainable AI for prediction interpretation
- Deployment using Flask or FastAPI

---

## 👨‍💻 Author

**Mallela Lokesh Reddy**

B.Tech – Computer Science and Engineering

---

## 📄 License

This project is intended for educational and research purposes.
