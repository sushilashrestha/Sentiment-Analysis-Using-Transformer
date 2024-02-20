# app.py
import streamlit as st
import torch
from encoder import Encoder
from dataset_preprocessor import DataPreprocessor
import re
#import nltk 
#from nltk.tokenize import word_tokenize
import string
from nepalitokenizers import WordPiece



tokenizer = WordPiece()



# Define function to load the model
# def load_model(model):
#     # Instantiate the model
#     model = Encoder(vocab_size=100, output_size=2, max_seq_len=512)

#     # Load the model weights
#     model_state_dict = torch.load(model)
#     model.load(model_state_dict)

#     # Set the model in evaluation mode
#     model.eval()

#     return model

# # Load the model
# model= '.\\mymodel'
# model = torch.load(model)

# # Function to perform prediction
# def predict(self,input_text,model):
#     # Process input text as needed (e.g., tokenization, conversion to tensors)
#     preprocess_dataset = DataPreprocessor(self.vocab_path)
#         # Remove HTML tag from review.
#     clean = re.compile('<.*?>')
#     review_without_tag = re.sub(clean, '', input_text)       
#         # Tokenize and remove punctuation from words.
#     review_without_punctuation = [''.join(char for char in word if (char not in string.punctuation)) for word in word_tokenize(review_without_tag)]
#         # Filter out empty strings.
#     filtered = list(filter(None, review_without_punctuation))
#         # Combine words into a sentence.
#     cleaned_sentence = ' '.join(filtered)
#         # Tokenize the cleaned input.
#     tokenized_sentence = preprocess_dataset.tokenize(cleaned_sentence)
#         # Create padding mask.
#     padding_mask = [0 if t == 0 else 1 for t in tokenized_sentence]
#         # Convert the tokenized sentence to a tensor and add batch dimension.
#     tokenized_input = torch.tensor(tokenized_sentence, dtype=torch.long).unsqueeze(0)
#         # Convert the padding mask into a torch tensor data type and adjust the size of the tensor to match the attention size.
#     padding_mask = torch.tensor(padding_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
#     # Perform prediction
#     with torch.no_grad():
#         # Forward pass
#         output = model(input_text)

#     # Process the output as needed
#     return output

# # Streamlit app
# def main():
#     st.title("Model Prediction")

#     # Add a text input for user to enter text
#     input_text = st.text_input("Enter some text:")

#     if st.button("Predict"):
#         # When the user clicks the Predict button, make the prediction
#         prediction = predict(input_text,model)
#         st.write(f"Prediction: {prediction}")

# if __name__ == "__main__":
#     main()


# Function to load the PyTorch model
def load_model(model_path):
    model = Encoder(vocab_size=100, output_size=2, max_seq_len=512)  # Initialize your model
    model.load(torch.load(model_path, map_location=torch.device('cpu')))  # Load model weights
    model.eval()  # Set model to evaluation mode
    return model

# Load the model
model='.\\mymodel'
model = torch.load(model)

# Function to perform prediction
# Function to perform prediction
def predict(input_text, model, tokenizer):
    # Process input text as needed (e.g., tokenization, conversion to tensors)
    vocab_path = ".\\dataset_vocab.model"
    preprocess_data = DataPreprocessor(vocab_path)
        # Remove HTML tag from review.
    clean = re.compile('<.*?>')
    review_without_tag = re.sub(clean, '', input_text)
        # Tokenize the review using WordPiece tokenizer.
    review_tokens = tokenizer.encode(review_without_tag).tokens
        # Remove punctuation from the tokens.
    review_without_punctuation = [''.join(char for char in word if (char not in string.punctuation)) for word in review_tokens]
        # Filter out empty strings.
    filtered = list(filter(None, review_without_punctuation))
        # Combine words into a sentence.
    cleaned_sentence = ' '.join(filtered)
        # Tokenize the cleaned input.
    tokenized_sentence = preprocess_data.tokenize(cleaned_sentence)
        # Create padding mask.
    padding_mask = [0 if t == 0 else 1 for t in tokenized_sentence]
        # Convert the tokenized sentence to a tensor and add batch dimension.
    tokenized_input = torch.tensor(tokenized_sentence, dtype=torch.long).unsqueeze(0)
    # Convert the padding mask into a torch tensor data type and adjust the size of the tensor to match the attention size.
    padding_mask = torch.tensor(padding_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        pred = model.forward(tokenized_input.to("cuda:0"), padding_mask.to("cuda:0"))
        scores = model.softmax(pred)

        print(f"Model's Predictions: {scores}\n   Positive: {scores[0,1]}\n   Negative: {scores[0,0]}")

    # Process the output as needed
    return scores

# Streamlit app
def main():
    st.title("Model Prediction")

    # Load the model
    model_path = '.\\mymodel'
    model = load_model(model_path)
    tokenizer = WordPiece()

    # Add a text input for user to enter text
    input_text = st.text_input("Enter some text:")

    if st.button("Predict"):
        # When the user clicks the Predict button, make the prediction
        prediction = predict(input_text, model, tokenizer)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()


# Streamlit app
def main():
    st.title("Model Prediction")

    # Add a text input for user to enter text
    input_text = st.text_input("Enter some text:")

    if st.button("Predict"):
        # When the user clicks the Predict button, make the prediction
        prediction = predict(input_text,model)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()