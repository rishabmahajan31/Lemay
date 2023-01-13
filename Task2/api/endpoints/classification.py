
from flask import Flask, jsonify, request, Blueprint
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import json

classification_api = Blueprint('classification_api', __name__) 


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

@app.route("/classification", methods=['GET'])
def classification():
    sample = json.loads(request.data)["text"]
    return jsonify(str(pipe(sample)[0]))

