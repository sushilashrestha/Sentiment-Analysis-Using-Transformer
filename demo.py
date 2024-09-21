from flask import Flask, request, jsonify, render_template
import torch
from encoder import Encoder
from nepalitokenizers import WordPiece
import re
import string

app = Flask(__name__)

# Load the model
model_path = "./mymodel"  # Update this to your model's path
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

tokenizer = WordPiece()

def preprocess_and_predict(sentence):
    # Remove HTML tags
    clean = re.compile('<.*?>')
    review_without_tag = re.sub(clean, '', sentence)
    
    # Tokenize and remove punctuation
    review_tokens = tokenizer.encode(review_without_tag).tokens
    review_without_punctuation = [''.join(char for char in word if char not in string.punctuation) for word in review_tokens]
    filtered = list(filter(None, review_without_punctuation))
    cleaned_sentence = ' '.join(filtered)
    
    # Tokenize the cleaned input
    tokenized_sentence = model.tokenize(cleaned_sentence)
    
    # Create padding mask
    padding_mask = [0 if t == 0 else 1 for t in tokenized_sentence]
    
    # Convert to tensor and add batch dimension
    tokenized_input = torch.tensor(tokenized_sentence, dtype=torch.long).unsqueeze(0)
    padding_mask = torch.tensor(padding_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(tokenized_input, padding_mask)
        scores = model.softmax(pred)
    
    return scores[0].tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.json['sentence']
    scores = preprocess_and_predict(sentence)
    return jsonify({'positive': scores[1], 'negative': scores[0]})

if __name__ == '__main__':
    app.run(debug=True)