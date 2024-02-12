# importing necessary modules
import string
import re
from nltk.tokenize import word_tokenize
import os
import time
import json
import sentencepiece as spm
from nepalitokenizers import WordPiece



class DatasetPreparer():

    def __init__(self, dataset_folder, tokenizer: WordPiece):
        self.tokenizer = tokenizer
        self.dataset_folder = dataset_folder

    """
    A Utility class for processing the dataset folder, preparing the dataset and buliding vocabulary.

    Args:
        dataset_folder (str): The path to the folder containing the extracted data files.

    Methods:
        preprocess(review: str) -> str:
            Preprocesses a given review by removing HTML tags, converting to lowercase,
            tokenizing, and removing punctuation.

        load_file(filename: str) -> str:
            Loads the content of a file into memory and returns it as a string.

        prepare_data_dict(directory: str, is_train: bool) -> dict:
            Prepares a dictionary of labeled reviews from the given directory for training or testing which contains reviews in the form of txt files.

        prepare_data_jsn(train_jsn_path: str, test_jsn_path: str) -> Tuple[str, str]:
            Prepares and saves the dataset dictionary as JSON files

        json_loader(train_json_path: str, test_json_path: str) -> Tuple[dict, dict]:
            Loads the dataset dictionaries from JSON files.

        prepare_dataset(
            train_json_path: str, test_json_path: str,
            verbose: bool = True, train_data_percentage: float = 0.9
        ) -> Tuple[list, list, list, list]:
            Prepares training and testing data by splitting and organizing reviews.

        write_to_file(file_path: str, sentences: List[str]) -> str:
            Writes cleaned sentences to a file for vocabulary building.

        build_vocab(
            vocab_file_prefix: str = 'imdb_vocab',
            vocab_size: int = 8000,
            model_type: str = 'bpe'
        ) -> str:
            Builds a vocabulary for encoding using SentencePiece.
    """

    def preprocess(self, review):
        # Remove HTML tags from the review.
        clean = re.compile('<.*?>')
        review_without_tag = re.sub(clean, '', review)
        # Convert the review to lowercase.
        review_lowercase = review_without_tag.lower()
        # Tokenize the review using WordPiece tokenizer.
        review_tokens = self.tokenizer.encode(review_lowercase).tokens
        # Remove punctuation from the tokens.
        review_without_punctuation = [''.join(char for char in word if (char not in string.punctuation)) for word in review_tokens]
        # Filter of Empty Strings
        filtered = list(filter(None, review_without_punctuation))
        #Combine the processed words into a sentence.
        return ' '.join(filtered)


    def load_file(self, filename):

        """
        Loads the content of a file into memeory and returns it as a string.

        Args:
            filename (str): The path to the file to be loaded.

        Returns:
            str: The content of the file.
        """

        # Open the file as read only
        file = open(filename, 'r', encoding = 'utf-8', errors ='replace')
        # Read all text
        text = file.read()
        # close the file
        file.close() # We can also do this within context using with statement which automatically close the file when we exist.

        return text


    def prepare_data_dict(self, directory, is_train):

        """
        Prepares a dictionary of labeled reviews from the given directory for training or testing.

        Args:
            directory (str): The directory containing the review files.
            is_train (bool): Whether preparing data for training or testing. True for Training and vice versa.

        Returns:
            dict: A dictionary containing positive and negative reviews.
        """

        review_dict={'neg':[],'pos':[]}

        if is_train:
            directory = os.path.join(directory+'/train')
        else:
            directory = os.path.join(directory+'/test')
        print('Directory : ',directory)

        for label_type in ['neg', 'pos']:

                data_folder=os.path.join(directory, label_type)
                print('Data Folder : ',data_folder)

                for root, dirs, files in os.walk(data_folder):
                    for fname in files:
                        print(fname)

                        if fname.endswith(".txt"):

                            file_name_with_full_path = os.path.join(root, fname)
                            review = self.load_file(file_name_with_full_path)
                            clean_review = self.preprocess(review)

                            if label_type == 'neg':
                                review_dict['neg'].append(clean_review)
                            else:
                                review_dict['pos'].append(clean_review)

        return review_dict


    def prepare_data_jsn(self, train_jsn_path, test_jsn_path):

        """
        Prepares and saves the dataset dictionary as JSON files.

        Args:
            train_jsn_path (str): Path to the training JSON file to write the training dict.
            test_jsn_path (str): Path to the testing JSON file to wite the test dict.

        Returns:
            Tuple[str, str]: Paths to the updated training and testing JSON files.
        """

        print("Checking whether dataset (train and test json) files already exist...")
        if not os.path.isfile(train_jsn_path) and not os.path.isfile(test_jsn_path):

            print("Dataset Files Does not exist")
            print("Creating train and test json files...")

            start = time.time()

            train_dict = self.prepare_data_dict(self.dataset_folder, is_train=True)
            test_dict = self.prepare_data_dict(self.dataset_folder, is_train=False)

            print("Successfully Created Dataset Dictionary. Now Saving them as Json files...")

            if len(train_dict['neg']) == len(train_dict['pos']) == len(test_dict['neg']) == len(test_dict['pos']):
                with open(train_jsn_path, 'w') as train_json_file:
                    json.dump(train_dict, train_json_file, indent=4)

                with open(test_jsn_path, 'w') as test_json_file:
                    json.dump(test_dict, test_json_file, indent=4)

                print("Sucessfully Saved the Dictionaries as Json files.")

            else:
                print("There's a Problem in the Dataset Folder. Check if it is Extracted Properly.")

            end = time.time()
            total_time = (end-start)/60
            print("Time Taken for Creating Dataset Files : ", total_time, "minutes")

            return train_jsn_path, test_jsn_path

        else:

            print("Both train and test json files Already Exist!")

            return train_jsn_path, test_jsn_path


    def json_loader(self, train_json_path, test_json_path):

        """
        Loads the dataset dictionaries from JSON files.

        Args:
            train_json_path (str): Path to the training JSON file.
            test_json_path (str): Path to the testing JSON file.

        Returns:
            Tuple[dict, dict]: Loaded training and testing datasets.
        """

        with open(train_json_path, 'r') as train_jsn:
            train = json.load(train_jsn)

        with open(test_json_path, 'r') as test_jsn:
            test = json.load(test_jsn)

        return train, test

    def prepare_dataset(self, train_json_path, test_json_path, verbose=True, train_data_percentage=0.9):

        """
        Prepares training and testing data by splitting and organizing reviews.

        Args:
            train_json_path (str): Path to the training JSON file.
            test_json_path (str): Path to the testing JSON file.
            verbose (bool): Whether to print the intermdiate information. Default is True.
            train_data_percentage (float): Percentage of data to use for training. Default is 0.9.

        Returns:
            Tuple[list, list, list, list]: Training and Testing dataset and their corresponding labels.
        """

        print("Preparing Training and Testing Data...")
        train_json_path, test_json_path = self.prepare_data_jsn(train_json_path, test_json_path)

        train_json, test_json = self.json_loader(train_json_path, test_json_path)

        if verbose:
            print()
            print("Number of Reviews in Training json file : ", len(train_json["pos"])+len(train_json['neg']))
            print('Number of Negative Reviews in Training json file :',len(train_json['neg']))
            print('Number of Positive Reviews in Training json file :',len(train_json['pos']))
            print()
            print("Number of Reviews in the Testing json file : ", len(test_json['neg'])+len(test_json['pos']))
            print('Number of Negative Reviews in Testing json file :',len(test_json['neg']))
            print('Number of Positive Reviews in Testing json file :',len(test_json['pos']))

        self.all_pos = train_json["pos"] + test_json['pos']
        self.all_neg = train_json['neg'] + test_json['neg']

        if verbose:
            print()
            print("Total Number of Positive Reviews : ", len(self.all_pos))
            print("Total Number of Negative Reviews : ", len(self.all_neg))

        train_data_len = int(train_data_percentage*(len(train_json["pos"])+len(train_json['neg'])))

        train_pos = self.all_pos[:train_data_len]
        train_neg = self.all_neg[:train_data_len]

        test_pos = self.all_pos[train_data_len:]
        test_neg = self.all_neg[train_data_len:]

        if verbose:
            print()
            print("Total Number of Positive in Training Set: ",len(train_pos))
            print("Total Number of Negative in Training Set: ",len(train_neg))
            print()
            print("Total Number of Positive in Testing Set: ",len(test_pos))
            print("Total Number of Negative in Testing Set: ",len(test_neg))

        train_x = train_pos + train_neg
        test_x = test_pos + test_neg

        train_y = [1] * len(train_pos) + [0] * len(train_neg)
        test_y = [1] * len(test_pos) + [0] * len(test_neg)

        if verbose:
            print()
            print("Number of Training x : ", len(train_x))
            print("Number of Training Target y : ", len(train_y))
            print()
            print("Number of Testing x : ", len(test_x))
            print("Number of Testing Target y : ", len(test_y))

        print("\nCreated Training and Testing Examples Sucessfully and Splitted according to the given percentage")

        return train_x, train_y, test_x, test_y

    def write_to_file(self, file_path, sentences):

        """
        Writes cleaned sentences to a file to build vocabulary on.

        Args:
            file_path (str): Path to the file to be written.
            sentences (List[str]): List of sentences to be written.

        Returns:
            str: Path to the written file.
        """

        print(f"Writing All Cleaned data to {file_path} for training vocabulary using sentencepiece")
        with open(file_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        print("Done!")
        return file_path


    def build_vocab(self, vocab_file_prefix='dataset_vocab', vocab_size=78, model_type='bpe'):

        """
        Builds a vocabulary using SentencePiece for encoding the reviews into numerical values.

        Args:
            vocab_file_prefix (str): Prefix for vocabulary file. Default is 'imdb_vocab'.
            vocab_size (int): Vocabulary size. Default is 8000. Other Options: 16000, 32000
            model_type (str): SentencePiece model type. Default is 'bpe'. other options: 'unigram', 'char', 'word'

        Returns:
            str: Path to the built vocabulary model file.
        """
        if type(self.dataset_folder) is tuple:
            self.dataset_folder = str(self.dataset_folder[0])
        dataset_parent_dir = os.path.dirname(self.dataset_folder)

        print("\nChecking whether the vocabulary file already exist...")
        vocab_file_path = os.path.join(dataset_parent_dir, vocab_file_prefix + '.model')
        if not os.path.isfile(vocab_file_path):

            data_txt_path = self.write_to_file(os.path.join(dataset_parent_dir, 'dataset_cleaned.txt'), self.all_pos+self.all_neg)

            if not os.path.isfile(data_txt_path):
                raise FileNotFoundError(f"Dataset file '{data_txt_path}' not found.")

            print("Creating Vocabulary using Sentencepiece as it doesn't already exist...")
            spm.SentencePieceTrainer.train(input=data_txt_path, model_prefix=vocab_file_prefix, vocab_size=vocab_size, model_type=model_type, pad_id=0, eos_id=1, unk_id=2, bos_id=-1, eos_piece='<EOS>', pad_piece='<PAD>', unk_piece='<UNK>')
            print("Vocabulary Created Sucessfully!")

            return vocab_file_path

        else:

            print("Vocabulary File Already Exist! Won't Train an other.")

            return vocab_file_path
        

        