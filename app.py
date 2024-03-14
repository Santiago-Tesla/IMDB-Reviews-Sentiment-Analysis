# app.py

from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

model_path = "my_awesome_model/checkpoint-126"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
    confidence = logits.softmax(dim=1).max().item()
    return sentiment, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sentiment, confidence = predict_sentiment(text)
    return jsonify({'sentiment': sentiment, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
