import json
import os
from typing import List

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


MODEL_DIR = "final_model"
LABELS_FILE = os.path.join(MODEL_DIR, "labels.json")

# Optional friendly display names for labels so output looks like
# "AI", "Computer Vision", "Machine Learning", etc.
FRIENDLY_LABELS = {
    "cs.CV": "Computer Vision / AI",
    "cs.LG": "Machine Learning / AI",
    "stat.ML": "Statistics / Machine Learning",
}


class PredictRequest(BaseModel):
    abstract: str


class ScoreItem(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    label: str
    score: float
    all_scores: List[ScoreItem]


app = FastAPI(title="Abstract Classifier API")


def load_model_and_tokenizer():
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(
            f"Model directory '{MODEL_DIR}' not found. "
            "Run train_model.py first to create final_model/."
        )
    if not os.path.isfile(LABELS_FILE):
        raise RuntimeError(
            f"Labels file '{LABELS_FILE}' not found. "
            "Ensure training finished successfully."
        )

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, labels, device


tokenizer, model, LABELS, DEVICE = load_model_and_tokenizer()


@app.get("/", response_class=HTMLResponse)
def index():
    # Modern, styled web UI to classify a single academic abstract
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Transformer-Based Abstract Classifier</title>
      <style>
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }
        
        body {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          background: linear-gradient(135deg, #00d4ff 0%, #5b86e5 25%, #ff6b9d 50%, #ffa500 75%, #00d4ff 100%);
          background-size: 400% 400%;
          animation: gradientShift 15s ease infinite;
          min-height: 100vh;
          padding: 20px;
          display: flex;
          justify-content: center;
          align-items: center;
        }
        
        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        
        .container {
          background: rgba(255, 255, 255, 0.98);
          backdrop-filter: blur(10px);
          border-radius: 25px;
          box-shadow: 0 25px 80px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.3);
          max-width: 900px;
          width: 100%;
          padding: 45px;
          animation: fadeIn 0.6s ease-in;
          border: 2px solid rgba(255, 255, 255, 0.5);
        }
        
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-20px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        
        h1 {
          color: #333;
          font-size: 2.4em;
          margin-bottom: 12px;
          text-align: center;
          background: linear-gradient(135deg, #00d4ff 0%, #5b86e5 30%, #ff6b9d 60%, #ffa500 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          background-size: 200% 200%;
          animation: textGradient 3s ease infinite;
          font-weight: 800;
          letter-spacing: -0.5px;
        }
        
        @keyframes textGradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        
        .subtitle {
          color: #555;
          text-align: center;
          margin-bottom: 35px;
          font-size: 1.15em;
          font-weight: 500;
        }
        
        .input-section {
          margin-bottom: 28px;
        }
        
        textarea {
          width: 100%;
          height: 180px;
          padding: 18px;
          border: 3px solid #e0e0e0;
          border-radius: 15px;
          font-size: 15px;
          font-family: inherit;
          resize: vertical;
          transition: all 0.4s ease;
          background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
          box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        textarea:focus {
          outline: none;
          border-color: #00d4ff;
          background: white;
          box-shadow: 0 0 0 4px rgba(0, 212, 255, 0.15), inset 0 2px 8px rgba(0, 0, 0, 0.05);
          transform: scale(1.01);
        }
        
        .button-container {
          text-align: center;
          margin-top: 25px;
        }
        
        button {
          background: linear-gradient(135deg, #00d4ff 0%, #5b86e5 50%, #ff6b9d 100%);
          background-size: 200% 200%;
          animation: buttonGradient 3s ease infinite;
          color: white;
          border: none;
          padding: 16px 50px;
          font-size: 17px;
          font-weight: 700;
          border-radius: 15px;
          cursor: pointer;
          transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
          box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4), 0 4px 10px rgba(91, 134, 229, 0.3);
          text-transform: uppercase;
          letter-spacing: 1px;
          position: relative;
          overflow: hidden;
        }
        
        @keyframes buttonGradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        
        button::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
          transition: left 0.5s;
        }
        
        button:hover::before {
          left: 100%;
        }
        
        button:hover {
          transform: translateY(-3px) scale(1.05);
          box-shadow: 0 12px 35px rgba(0, 212, 255, 0.5), 0 6px 15px rgba(91, 134, 229, 0.4);
        }
        
        button:active {
          transform: translateY(-1px) scale(1.02);
        }
        
        button:disabled {
          opacity: 0.7;
          cursor: not-allowed;
          transform: none;
        }
        
        .result {
          margin-top: 35px;
          padding: 30px;
          border-radius: 20px;
          background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(91, 134, 229, 0.1) 50%, rgba(255, 107, 157, 0.1) 100%);
          border: 2px solid rgba(0, 212, 255, 0.3);
          animation: slideDown 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
          display: none;
          backdrop-filter: blur(5px);
        }
        
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-15px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        
        .result.show {
          display: block;
        }
        
        .prediction {
          font-size: 1.4em;
          color: #2c3e50;
          margin-bottom: 25px;
          padding: 20px;
          background: linear-gradient(135deg, #ffffff 0%, #f0f8ff 100%);
          border-radius: 15px;
          border-left: 6px solid;
          border-image: linear-gradient(135deg, #00d4ff, #ff6b9d) 1;
          box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
          font-weight: 600;
        }
        
        .prediction strong {
          background: linear-gradient(135deg, #00d4ff 0%, #5b86e5 50%, #ff6b9d 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          font-weight: 800;
        }
        
        .score {
          color: #ff6b9d;
          font-weight: 700;
          text-shadow: 0 2px 4px rgba(255, 107, 157, 0.2);
        }
        
        .scores-table {
          margin-top: 25px;
          overflow-x: auto;
        }
        
        table {
          width: 100%;
          border-collapse: separate;
          border-spacing: 0;
          background: white;
          border-radius: 15px;
          overflow: hidden;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        
        thead {
          background: linear-gradient(135deg, #00d4ff 0%, #5b86e5 50%, #ff6b9d 100%);
          background-size: 200% 200%;
          animation: headerGradient 4s ease infinite;
          color: white;
        }
        
        @keyframes headerGradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        
        th {
          padding: 18px;
          text-align: left;
          font-weight: 700;
          text-transform: uppercase;
          font-size: 0.95em;
          letter-spacing: 1px;
          text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        td {
          padding: 15px 18px;
          border-bottom: 1px solid rgba(0, 212, 255, 0.1);
        }
        
        tr:last-child td {
          border-bottom: none;
        }
        
        tr:hover {
          background: linear-gradient(90deg, rgba(0, 212, 255, 0.08) 0%, rgba(91, 134, 229, 0.08) 100%);
          transition: all 0.3s ease;
          transform: scale(1.01);
        }
        
        .label-cell {
          font-weight: 700;
          color: #2c3e50;
        }
        
        .score-cell {
          color: #ff6b9d;
          font-weight: 700;
          font-size: 1.05em;
        }
        
        .loading {
          text-align: center;
          padding: 25px;
          background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(91, 134, 229, 0.1) 100%);
          border-radius: 15px;
          color: #00d4ff;
          font-size: 1.2em;
          font-weight: 600;
        }
        
        .loading::after {
          content: '...';
          animation: dots 1.5s steps(4, end) infinite;
        }
        
        @keyframes dots {
          0%, 20% { content: '.'; }
          40% { content: '..'; }
          60%, 100% { content: '...'; }
        }
        
        .error {
          color: #ff4757;
          background: linear-gradient(135deg, rgba(255, 71, 87, 0.1) 0%, rgba(255, 107, 157, 0.1) 100%);
          padding: 18px;
          border-radius: 12px;
          border-left: 5px solid #ff4757;
          box-shadow: 0 4px 15px rgba(255, 71, 87, 0.2);
          font-weight: 600;
        }
        
        @media (max-width: 768px) {
          .container {
            padding: 25px;
          }
          
          h1 {
            font-size: 1.8em;
          }
          
          textarea {
            height: 150px;
          }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Transformer-Based Classification for Academic Paper Abstracts</h1>
        <p class="subtitle">Enter an academic abstract below and click <strong>Classify</strong> to see the predicted category and scores.</p>
        
        <div class="input-section">
          <textarea id="abstractInput" placeholder="Type or paste an academic abstract here..."></textarea>
        </div>
        
        <div class="button-container">
          <button id="classifyBtn" onclick="classify()">Classify Abstract</button>
        </div>
        
        <div id="result" class="result"></div>
      </div>

      <script>
        async function classify() {
          const abstractText = document.getElementById('abstractInput').value.trim();
          const resultDiv = document.getElementById('result');
          const btn = document.getElementById('classifyBtn');
          
          if (!abstractText) {
            resultDiv.className = 'result show';
            resultDiv.innerHTML = '<div class="error"><strong>⚠️ Please enter an abstract.</strong></div>';
            return;
          }
          
          btn.disabled = true;
          btn.textContent = 'Classifying...';
          resultDiv.className = 'result show';
          resultDiv.innerHTML = '<div class="loading">Classifying</div>';
          
          try {
            const resp = await fetch('/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ abstract: abstractText })
            });
            
            if (!resp.ok) {
              throw new Error(resp.statusText);
            }
            
            const data = await resp.json();
            
            let html = '<div class="prediction">';
            html += '<strong>The abstract is classified as:</strong> ';
            html += '<span style="background: linear-gradient(135deg, #00d4ff 0%, #5b86e5 50%, #ff6b9d 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 800; font-size: 1.1em;">' + data.label + '</span>';
            html += ' <span class="score">(score: ' + data.score.toFixed(4) + ')</span>';
            html += '</div>';
            
            if (data.all_scores && data.all_scores.length) {
              html += '<div class="scores-table">';
              html += '<table><thead><tr><th>Label</th><th>Confidence Score</th></tr></thead><tbody>';
              for (const item of data.all_scores) {
                html += '<tr>';
                html += '<td class="label-cell">' + item.label + '</td>';
                html += '<td class="score-cell">' + item.score.toFixed(4) + '</td>';
                html += '</tr>';
              }
              html += '</tbody></table></div>';
            }
            
            resultDiv.innerHTML = html;
          } catch (err) {
            resultDiv.innerHTML = '<div class="error"><strong>❌ Error:</strong> ' + err.message + '</div>';
          } finally {
            btn.disabled = false;
            btn.textContent = 'Classify Abstract';
          }
        }
        
        // Allow Enter key to submit (Ctrl+Enter)
        document.getElementById('abstractInput').addEventListener('keydown', function(e) {
          if (e.ctrlKey && e.key === 'Enter') {
            classify();
          }
        });
      </script>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels")
def get_labels():
    return LABELS


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.abstract.strip()
    if not text:
        return PredictResponse(label="", score=0.0, all_scores=[])

    with torch.no_grad():
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        outputs = model(**encoded)
        logits = outputs.logits.detach().cpu().numpy()[0]
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()

    all_scores = []
    for i in range(len(LABELS)):
        raw_label = LABELS[i]
        friendly = FRIENDLY_LABELS.get(raw_label, raw_label)
        all_scores.append(ScoreItem(label=friendly, score=float(probs[i])))
    all_scores_sorted = sorted(all_scores, key=lambda x: x.score, reverse=True)
    top = all_scores_sorted[0]

    return PredictResponse(label=top.label, score=top.score, all_scores=all_scores_sorted)




