# Importing All the Modules that we created
import yaml # We Are Going to use yaml to parse the Arguments. Forget the Good old Arg Parser >_<
from helpers import model_summary, count_parameters, plot_metrics
import torch
from dataset_initializer import DataInitializer
from dataset_preparer import DatasetPreparer
from dataset_preprocessor import DataPreprocessor
from batch_iterator import BatchIterator
from trainer import Trainer
import nltk 
from nltk.tokenize import word_tokenize
import re
import string
import nltk
nltk.download('punkt')
from nepalitokenizers import WordPiece
from encoder import Encoder

tokenizer = WordPiece()

class Prepare_Train():

    def __init__(self, dataset_dir, train_json_path, test_json_path, batch_size=4, output_size=2, 
                 max_seq_len=512, d_model=512, pooling="max", num_heads=4, expansion_factor=2, 
                 num_blocks=2, activation="relu", dropout_size=0.1, model_save_path=None, 
                 criterator="cel", optimizer_type="adamw", num_epochs=5, learning_rate=1e-3, 
                 weight_decay=1e-5, gamma=0.1):

        """
        Class which Encapsulate all the Necessary Modules to Train a Sentiment Analysis Model From Scratch.

        Args:
            dataset_dir (str): Path to the dataset directory.
            train_json_path (str): Path to the training JSON file.
            test_json_path (str): Path to the test JSON file.
            batch_size (int): Batch size for training.
            output_size (int): Number of output classes.
            max_seq_len (int): Maximum sequence length.
            d_model (int): Dimension of the model.
            pooling (str): Pooling method to downsample. Default is "max".
            num_heads (int): Number of attention heads. Default is 4.
            expansion_factor (int): Expansion factor for feedforward layer. Default is 2.
            num_blocks (int): Number of encoder blocks. Default is 2.
            activation (str): Activation function. Default is "relu".
            dropout_size (float): Dropout rate. Default is 0.1.
            model_save_path (str): Path to save the trained model. Default is None.
            criterator (str): Loss Function Algorithm. Default is "cel". Other Options: "bce"
            optimizer_type (str): Optimizing Algorithm used in training. Default is "adamw". Other Options: "radam"
            num_epochs (int): Number of Epochs to train the model for. Default is 5
            learning_rate (float): Learning rate for gradient calculation. Default is 1e-3.
            weight_decay (float): Weight decay (L2 Normalization). Default is 1e-5.
            gamma (float): gamma factor for learning rate scheduler. Default is 0.1
        """

        self.dataset_dir = dataset_dir
        self.train_json_path = train_json_path
        self.test_json_path = test_json_path
        self.batch_size = batch_size
        self.output_size = output_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pooling = pooling
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        self.num_blocks = num_blocks
        self.activation = activation
        self.dropout_rate = dropout_size
        self.model_save_path = model_save_path
        self.criterator = criterator
        self.optimizer_type = optimizer_type
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma

    def initialize_the_dataset(self):
    
        """Initialize the Dataset directory and prepare dataset folder."""
        
        initialize_dataset = DataInitializer(self.dataset_dir)
        self.dataset_folder = initialize_dataset.prepare_dataset_folder()

    def prepare_the_dataset(self):

        """Prepare the dataset for training and validation also build vocabulary to encode the input"""
    
        dataset_prep = DatasetPreparer(self.dataset_folder, tokenizer = tokenizer)
        self.train_x, self.train_y, self.test_x, self.test_y = dataset_prep.prepare_dataset(self.train_json_path, self.test_json_path, verbose=True)
        self.vocab_path =  dataset_prep.build_vocab()

    def preprocess_the_dataset(self):

        """Preprocess the dataset by Filtering and Load the Vocabulary to use externaly."""
        
        preprocess_dataset = DataPreprocessor(self.vocab_path)
        self.vocab = preprocess_dataset.load_vocab(self.vocab_path)
        self.input_vocab_size = self.vocab.get_piece_size()
        self.pad_idx = self.vocab.piece_to_id('<PAD>')
        preprocess_dataset.check(self.train_x, self.train_y, self.test_x, self.test_y, before=True)
        self.filtered_train_x, self.filtered_train_y, self.filtered_test_x, self.filtered_test_y = preprocess_dataset.filter_by_length(self.train_x, self.train_y, self.test_x, self.test_y)
        preprocess_dataset.check(self.filtered_train_x, self.filtered_train_y, self.filtered_test_x, self.filtered_test_y, before=False)
        self.tokenize = preprocess_dataset.tokenize

    def initialize_the_iterator(self):

        """Initialize data iterators for training and testing."""

        iterator = BatchIterator(self.filtered_train_x, self.filtered_train_y, self.filtered_test_x, self.filtered_test_y, self.tokenize)
        self.train_generator = iterator.data_generator(self.batch_size, train=True)
        self.test_generator = iterator.data_generator(self.batch_size, train=False)
        self.batch_per_epoch_train, self.batch_per_epoch_test = iterator.calculate_batch_per_epoch(self.batch_size)

    def predict(self, sentence, model, tokenize):
        """
    Predict the emotion of the given sentence using the trained model.

    Args:
        sentence (str): Input sentence to predict sentiment.
        model: Trained sentiment analysis model.
        tokenize: Tokenization function.

    Prints:
        Model's Prediction of given sentence being positive and negative
    """

        preprocess_dataset = DataPreprocessor(self.vocab_path)
        # Remove HTML tag from review.
        clean = re.compile('<.*?>')
        review_without_tag = re.sub(clean, '', sentence)
        # Make Entire Sentence lowercase.
        review_lowercase = review_without_tag.lower()
        # Tokenize and remove punctuation from words.
        review_without_punctuation = [''.join(char for char in word if (char not in string.punctuation)) for word in word_tokenize(review_lowercase)]
        # Filter out empty strings.
        filtered = list(filter(None, review_without_punctuation))
        # Combine words into a sentence.
        cleaned_sentence = ' '.join(filtered)
        # Tokenize the cleaned input.
        tokenized_sentence = preprocess_dataset.tokenize(cleaned_sentence)
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

    def train_the_model(self):

        """Train a Sentiment Analysis Model from scratch (without pretrained word embeddings) using Prepared Dataset so far

        Prints:
            Comprehensive Summary of the Model and It's Parameter
        
        Plots:
            Metrics of the Model's Loss and Accuracy while training and testing
            
        Returns:
            Path to the Saved Model
        
        """

        trainer = Trainer(
            self.train_generator,
            self.batch_per_epoch_train,
            self.test_generator,
            self.batch_per_epoch_test,
            self.input_vocab_size,
            self.output_size,
            self.max_seq_len,
            self.d_model,
            pad_ix=self.pad_idx,
            pooling=self.pooling,
            num_heads=self.num_heads,
            expansion_factor=self.expansion_factor,
            num_blocks=self.num_blocks,
            activation=self.activation,
            dropout_size=self.dropout_rate,
            model_save_path=self.model_save_path,
            criterator=self.criterator,
            optimizer_type=self.optimizer_type
        )
        
        self.prepared_model = trainer.prepare_model()
        model_summary(self.prepared_model, self.train_generator)
        count_parameters(self.prepared_model)
        self.train_loss, self.train_acc, self.test_loss, self.test_acc, model_path = trainer.train(self.prepared_model, num_epochs=self.num_epochs, learning_rate=self.learning_rate, weight_decay=self.weight_decay, gamma=self.gamma)
        #plot_metrics(self.num_epochs, self.batch_per_epoch_train, self.batch_per_epoch_test, self.train_loss, self.train_acc, self.test_acc)
        plot_metrics(self.train_loss, self.train_acc,self.test_loss, self.test_acc)
        
        return model_path


if __name__ == "__main__":

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

    # Execute necessary steps
    prepare_and_train.initialize_the_dataset()
    prepare_and_train.prepare_the_dataset()
    prepare_and_train.preprocess_the_dataset()
    prepare_and_train.initialize_the_iterator()
    model_path = prepare_and_train.train_the_model()

    #Test the trained model
# test = input("Do You Want to Test the Model Now (y/n): ")
# if test.lower() == "y":
#     print("Getting Ready for Inference... Enter `q` to exit.")
#     model_path = ".\\mymodel"
#     model_class = Encoder
#     model = model_class(vocab_size=100, output_size=2, max_seq_len=512)
#     state_dict = torch.load(model_path)

#     # Update the embedding.embed.weight parameter to match the size in the state dictionary
#     state_dict['embedding.embed.weight'] = state_dict['embedding.embed.weight'].unsqueeze(2).expand(-1, -1, 512)

#     model.load_state_dict(state_dict)
#     # model = prepare_and_train.prepared_model.load_state_dict(torch.load(model_path))
#     while True:
#         sentence = input("Enter your Sentence: ")
#         if sentence != "q":
#             prepare_and_train.predict(sentence, model)
#         break
    
    

    test = input("Do You Want to Test the Model Now (y/n): ")
    if test.lower() == "y":
        print("Getting Ready for Inference... Enter `q` to exit.")
        model = torch.load(model_path)
        print(model)
        while True:
            sentence = input("Enter your Sentence: ")
            if sentence != "q":
                prepare_and_train.predict(sentence, model, prepare_and_train.tokenize)
            break