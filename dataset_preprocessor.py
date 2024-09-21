import torch
import os
import sentencepiece as spm

class DataPreprocessor():

    def __init__(self,
                 vocab_path,
                 ):

        """
        A Utility class for preprocessing the dataset and preparing vocabulary for training.

        Args:
            vocab_path (str): Path to the SentencePiece vocabulary model.

        Methods:
            filter_by_length(
                train_x: list, train_y: list, test_x: list, test_y: list, max_seq_len: int = 512
            ) -> tuple:
                Filters data samples by sequence length, ensuring they don't exceed max_seq_len.

            check(
                train_x: list, train_y: list, test_x: list, test_y: list, before: bool = True
            ) -> None:
                Performs data consistency checks and prints statistics before or after filtering.

            load_vocab(vocab_model_path: str) -> SentencePieceProcessor:
                Loads the SentencePiece vocabulary processor from the given path.

            tokenize(line: str) -> list:
                Tokenizes a given line using the loaded vocabulary.

            detokenize(tokens: list) -> str:
                Converts a list of tokens back into a string using the loaded vocabulary.

            get_pos_onehot(length: int) -> torch.Tensor:
                Generates a one-hot tensor for positional encoding with the specified length.

        Attributes:
            vocab: Loaded SentencePiece vocabulary object.
            EOS: Token ID for end-of-sentence marker.
            PAD: Token ID for padding marker.
        """

        self.vocab_path = vocab_path
        self.vocab = self.load_vocab(self.vocab_path) # SentencePiece vocabulary
        self.EOS = self.vocab.piece_to_id('<EOS>') # Token ID to mark end-of-sentence
        self.PAD = self.vocab.piece_to_id('<PAD>') # Token ID for padding


    def filter_by_length(self, train_x, train_y, test_x, test_y, max_seq_len=512):

        """
        Filters data samples by maximum sequence length.

        Args:
            train_x (list): List of training data.
            train_y (list): List of training labels.
            test_x (list): List of testing data.
            test_y (list): List of testing labels.
            max_seq_len (int): Maximum sequence length the samples can have.

        Returns:
            tuple: Filtered training and testing data and their corresponding labels.
        """

        filtered_train_x = []
        filtered_train_y = []
        filtered_test_x = []
        filtered_test_y = []

        # [(filtered_train_x.append(review), filtered_train_y.append(sentiment)) for review, sentiment in zip(train_x, train_y)  if len(self.tokenize(review)) <= max_seq_len]
        # [(filtered_test_x.append(review), filtered_test_y.append(sentiment)) for review, sentiment in zip(test_x, test_y) if len(self.tokenize(review)) <= max_seq_len]


        for review, sentiment in zip(train_x, train_y):
            if len(self.tokenize(review)) <= max_seq_len:
                filtered_train_x.append(review)
                filtered_train_y.append(sentiment)

        for review, sentiment in zip(test_x, test_y):
            if len(self.tokenize(review)) <= max_seq_len:
                filtered_test_x.append(review)
                filtered_test_y.append(sentiment)

        return filtered_train_x, filtered_train_y, filtered_test_x, filtered_test_y


    def check(self, train_x, train_y, test_x, test_y, before=True):

        """
        Checks and prints dataset statistics.

        Args:
            train_x (list): List of training data.
            train_y (list): List of training labels.
            test_x (list): List of testing data.
            test_y (list): List of testing labels.
            before (bool): If True, displays statistics before filtering. If False, displays statistics after filtering.

        Raises:
            AssertionError: If training examples and their corresponding labels or testing examples and their corresponding labels have imbalanced lengths.

        Prints:
            Dataset statistics.
        """

        assert len(train_x) == len(train_y), "Imbalanced Training Set. length of train_x must be equal to length of train_y. Check if there's any problem with filtering process."
        assert len(test_x) == len(test_y), "Imbalanced Testing Set. length of test_x must be equal to length of test_y. Check if there's any problem with filtering process."

        train_max_length = max(len(self.tokenize(x)) for x in train_x)
        test_max_length = max(len(self.tokenize(x)) for x in test_x)
        total_tokens = sum(len(self.tokenize(x))-1 for x in train_x + test_x)

        if before:
            print("Maximum Length of the Review in Train set Before Filtering: ", train_max_length)
            print("Maximum Length of the Review in Test Set Before Filtering: ", test_max_length)
            print("Total Number of Tokens in the Training and Testing Set Before Filtering: ", total_tokens)
            print("Number of Reviews in Training Set Before Filtering: ", len(train_x))
            print("Number of Sentiments in Training Set Before Filtering: ", len(train_y))
            print("Number of Reviews in Testing Set Before Filtering: ", len(test_x))
            print("Number of Sentiments in Testing Set Before Filtering: ", len(test_y))
            print()

        else:
            print("Maximum Length of the Review in Train Set After Filtering: ", train_max_length)
            print("Maximum Length of the Review in Test Set After Filtering: ", test_max_length)
            print("Total Number of Tokens in the Training and Testing Set After Filtering: ", total_tokens)
            print("Number of Reviews in Training Set After Filtering: ", len(train_x))
            print("Number of Sentiments in Training Set After Filtering: ", len(train_y))
            print("Number of Reviews in Testing Set After Filtering: ", len(test_x))
            print("Number of Sentiments in Testing Set After Filtering: ", len(test_y))
            print()


    def load_vocab(self, vocab_model_path):

        """
        Loads the SentencePiece vocabulary from the given path.

        Args:
            vocab_model_path (str): Path to Sentencepiece Vocabulary File.

        Returns:
            Sentencepiece Vocabulary object
        """

        return spm.SentencePieceProcessor(model_file=vocab_model_path)


    def tokenize(self, line):

        """
        Tokenizes a given line using the loaded vocabulary.

        Args:
            line (str): Input text to tokenize.

        Returns:
            list: List of token IDs.
        """

        return self.vocab.encode(line)+[self.EOS]


    def detokenize(self, tokens):

        """
        Detokenizes a list of token IDs using the loaded vocabulary. We Doesn't Need this one for this project. Just Creating it, in case if we want to decode the input to the model.

        Args:
            tokens (list): List of token IDs.

        Returns:
            str: Detokenized text.
        """

        return self.vocab.decode(tokens)

    def get_pos_onehot(self, length):

        """
        Generates a one-hot matrix for positional encoding. This Function also isn't used in this project but you can use, incase if you want the input to be one hot encoded sequence instead of token ids.

        Args:
            length (int): Length of the sequence.

        Returns:
            torch.Tensor: One-hot matrix.
        """

        onehot = torch.zeros(length,length)
        idxs = torch.arange(length).long().view(-1,1)
        onehot.scatter_(1,idxs,1)
        return onehot