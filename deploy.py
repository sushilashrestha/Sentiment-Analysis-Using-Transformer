import streamlit as st
import torch

from encoder import Encoder  # Assuming this is your custom encoder class
from dataset_preprocessor import DataPreprocessor

import re
import string
from nepalitokenizers import WordPiece
from nepali_unicode_converter.convert import Converter


tokenizer = WordPiece()
# Function to load the PyTorch model
def load_model(model_path):
    model = Encoder(vocab_size=100, output_size=2, max_seq_len=512)  # Initialize your model
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Load model state dictionary
    model.load_state_dict(state_dict)  # Load model weights
    model.eval()  # Set model to evaluation mode
    return model





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

    # Process the output as needed
    return scores

# Streamlit app
def main():
    st.title("Model Prediction")

    # Add a text input for user to enter text
    input_text = st.text_input("Enter some text:")
    converter= Converter()
    translated_text =converter.convert(input_text)
    if st.button("Predict"):
        # Load the model
        model_path = '.\\mymodel'
        model = torch.load(model_path)
        
        # When the user clicks the Predict button, make the prediction
        scores = predict(input_text, model, tokenizer)
        
        # Display the prediction
        # st.write(f"Prediction: {scores}")
        if ({scores[0,1]}>{scores[0,0]}):
            st.write("SENTENCE IS POSITIVE")
        else:
            st.write("SENTENCE IS NEGATIVE")
        st.write(f"Model's Predictions:   Positive: {scores[0,1]}\n   Negative: {scores[0,0]}")


if __name__ == "__main__":
    main()
