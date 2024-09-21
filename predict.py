import re
import string
import torch
import yaml
import nltk

from nltk.tokenize import word_tokenize
from nepalitokenizers import WordPiece
from encoder import Encoder
from main import Prepare_Train
from dataset_preprocessor import DataPreprocessor

tokenizer = WordPiece()

def predict(input_text, model_path):
    """
    Predict the emotion of the given sentence using the trained model.

    Args:
        sentence (str): Input sentence to predict sentiment.
        model_path (str): Path to the trained model.

    Prints:
        Model's Prediction of given sentence being positive and negative
    """

    # Preprocess the sentencedef predict(input_text, model):
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
# Main function
def main():
    # Load arguments from a YAML config file
    with open("config.yaml", 'r') as file:
        try:
            args = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Initialize the training process
    prepare_and_train = Prepare_Train(
        args["DATASET_DIR"],
        args["TRAIN_JSON_PATH"],
        args["TEST_JSON_PATH"],
        batch_size=args["BATCH_SIZE"],
        output_size=args["OUTPUT_SIZE"],
        max_seq_len=args["MAX_SEQ_LEN"],
        d_model=args["EMBEDDING_DIMENSION"],
        pooling = args["POOLING_METHOD"],
        num_heads=args["NUM_HEADS"],
        expansion_factor=args["EXPANSION_FACTOR"],
        num_blocks=args["NUM_BLOCKS"],
        activation=args["ACTIVATION"],
        dropout_size=args["DROPOUT"],
        model_save_path=args["MODEL_SAVE_PATH"],
        criterator=args["LOSS_FN_TYPE"],
        optimizer_type=args["OPTIMIZER_TYPE"],
        num_epochs=args["NUM_EPOCHS"],
        learning_rate=args["LEARNING_RATE"],
        weight_decay=args["WEIGHT_DECAY"],
        gamma=args["GAMMA"]
    )

def load_model(model_path):
    model = Encoder(vocab_size=100, output_size=2, max_seq_len=512)  # Initialize your model
    model.load(torch.load(model_path, map_location=torch.device('cpu')))  # Load model weights
    model.eval()  # Set model to evaluation mode
    return model

# Load the model
model='.\\mymodel'
model = torch.load(model)
test = input("Do You Want to Test the Model Now (y/n): ")
if test.lower() == "y":
    print("Getting Ready for Inference... Enter `q` to exit.")
    model_path = ".\\mymodel"
    while True:
        sentence = input("Enter your Sentence: ")
        if sentence != "q":
            predict(sentence, model)
        else:
            break