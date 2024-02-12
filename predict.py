# Importing all the required modules
import re
import string
import torch
import yaml
import nltk
from nltk.tokenize import word_tokenize
from nepalitokenizers import WordPiece
from encoder import Encoder
from main import Prepare_Train
# Loading the tokenizer
tokenizer = WordPiece()

def predict(sentence, model_path):
    """
    Predict the emotion of the given sentence using the trained model.

    Args:
        sentence (str): Input sentence to predict sentiment.
        model_path (str): Path to the trained model.

    Prints:
        Model's Prediction of given sentence being positive and negative
    """

    # Preprocess the sentence
    clean = re.compile('<.*?>')
    review_without_tag = re.sub(clean, '', sentence)
    review_lowercase = review_without_tag.lower()
    review_without_punctuation = [''.join(char for char in word if (char not in string.punctuation)) for word in word_tokenize(review_lowercase)]
    filtered = list(filter(None, review_without_punctuation))
    cleaned_sentence = ' '.join(filtered)

    # Tokenize the cleaned input
    tokenized_sentence = tokenizer.encode(cleaned_sentence)

    # Convert the tokenized sentence to a list of IDs
    tokenized_ids = tokenized_sentence.ids

    # Create Padding Mask
    padding_mask = [0 if t == 0 else 1 for t in tokenized_ids]

    # Convert the tokenized sentence to a tensor and add batch dimension
    tokenized_input = torch.tensor(tokenized_ids, dtype=torch.long).unsqueeze(0)

    # Convert the Padding mask into a torch tensor data type and adjust the size of the tensor to match the attention size
    padding_mask = torch.tensor(padding_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # Load the trained model
    model_class = Encoder
    model = model_class(vocab_size=100, output_size=2, max_seq_len=512)

    # Load the state dictionary of the model
    state_dict = torch.load(model_path)

    # Update the embedding.embed.weight parameter to match the size in the state dictionary
    state_dict['embedding.embed.weight'] = state_dict['embedding.embed.weight'].unsqueeze(2).expand(-1, -1, 512)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    # Make a prediction
    with torch.no_grad():
        pred = model.forward(tokenized_input.to("cuda:0"), padding_mask.to("cuda:0"))
        scores = model.softmax(pred)

    print(f"Model's Predictions: {scores}\n   Positive: {scores[0,1]}\n   Negative: {scores[0,0]}")
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
test = input("Do You Want to Test the Model Now (y/n): ")
if test.lower() == "y":
    print("Getting Ready for Inference... Enter `q` to exit.")
    model_path = ".\\mymodel"
    while True:
        sentence = input("Enter your Sentence: ")
        if sentence != "q":
            predict(sentence, model_path)
        else:
            break