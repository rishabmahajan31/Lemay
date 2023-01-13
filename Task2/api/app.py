from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import json

app = Flask(__name__)  # define app using Flask

tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")

model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

@app.route("/classification")
def classification():
    args = request.args   
    sample = args.get("text", type=str)
    print(sample)
    #sample = json.loads(request.data)["text"]
    return jsonify(str(pipe(sample)))

@app.route('/')
def test():
    return jsonify({'message':'It works! '})


if __name__ == '__main__':
    app.run()
