import random as rnd
import torch

class BatchIterator():

    def __init__(self,
                 train_x,
                 train_y,
                 test_x,
                 test_y,
                 tokenize
                 ):
        
        """
        A Utility class for generating batches of data to feed the model for training.

        Args:
            train_x (list): List of training samples.
            train_y (list): List of corresponding sentiment labels for training data.
            test_x (list): List of testing samples.
            test_y (list): List of corresponding sentiment labels for testing data.
            tokenize (method): Function to tokenize the samples

        Methods:
            calculate_batch_per_epoch(batch_size: int) -> Tuple[int, int]:
                Calculates the number of batches per epoch for training and testing data.

            data_generator(batch_size: int, train: bool = True, shuffle: bool = True) -> Generator:
                Generates Infinte batches of data for training or testing.
        """

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.tokenize = tokenize
        self.PAD = 0

    def calculate_batch_per_epoch(self, batch_size):

        """
        Calculates the number of batches per epoch.

        Args:
            batch_size (int): Batch size.

        Returns:
            tuple: Number of training and testing batches per epoch.
        """

        train_batch_per_epoch = int(len(self.train_x) // batch_size)
        test_batch_per_epoch = int(len(self.test_x) // batch_size)

        return train_batch_per_epoch, test_batch_per_epoch


    def data_generator(self, batch_size, train=True, shuffle=True): 

        """
        Yields Infinte Number batches of data for training and testing. set `train=True` for yielding training set and vice versa. if we loop through the generator to yield batches of data,
        It will keep yielding forever. That's the Reason why we are calculating the number of batches per epoch to make sure the model sees every examples in the training set in one epoch.

        Args:
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle data. Default is True.

        Yields:
            tuple: Batches of reviews, their corresponding sentiments and their corresponding padding masks.
        """

        if train:
            x, y = self.train_x, self.train_y

        else:
            x, y = self.test_x, self.test_y

        # initialize the index that points to the current position in the lines index array
        index = 0

        # variable for storing the length of the longest sequence in every batch
        max_len = 0

        # initialize the list that will contain the review and thire corresponding label batch
        reviews = []
        sentiments = []

        # count the number of examples in x
        num_lines = len(x)

        # create an array with the indexes of x that can be shuffled
        lines_index = [*range(num_lines)]

        # print("Line Index Before : ", lines_index[:5])

        # shuffle line indexes if shuffle is set to True
        if shuffle:
            rnd.shuffle(lines_index)

        # print("Line Index After Shuffling : ", lines_index[:5])

        while True:

            # if the index is greater than or equal to the number of examples in x
            if index>=num_lines:
                # then reset the index to 0
                index = 0
                # shuffle line indexes if shuffle is set to True
                if shuffle:
                    rnd.shuffle(lines_index)

            # get a example at the `lines_index[index]` position in x and their corresponding label
            review = self.tokenize(x[lines_index[index]])
            sentiment = y[lines_index[index]]

            # capture the length of the current sequence
            lenx = len(review)

            # if the length of the current sequence is longer than longest sequence in the current batch, then set the max_len to current sequence length
            if lenx > max_len:
                max_len = lenx

            # print("Max Length Before : ",max_len)

            # append the sample and thier corresponding labels to the review and sentiment batchs
            reviews.append(review)
            sentiments.append(sentiment)

            index += 1

            # if the current batch size is equal to the desired batch size, then process and yield them.
            if len(reviews) == len(sentiments) == batch_size:

                # print("Final Max Length : ",max_len)

                review_batch = []
                padding_mask_batch = []

                # go through each review in review batch
                for review in reviews:

                    # Create a list of pad id to represent the padding
                    padding = [self.PAD] * (max_len - len(review))

                    # combine the review plus pad id list so that the review list plus padding will have length `max_len`
                    review_padded = review + padding

                    # print("Length of the Padded Tensor : ", len(tensor_pad))
                    # append the padded list to the review batch
                    review_batch.append(review_padded)

                    # creating padding mask for each review to be used in dot product attention to mask the padded input
                    padding_mask = [0 if token == 0 else 1 for token in review_padded]
                    padding_mask_batch.append(padding_mask)

                # convert the batch of data type list into torch tensors
                reviews_tensor = torch.tensor(review_batch, dtype=torch.long)
                sentiment_batch_tensor = torch.tensor(sentiments, dtype=torch.long)

                # Adding additional dimension along 2 and 3 axis to match the shape of the tensor when calculating attention.
                padding_batch_tensor = torch.tensor(padding_mask_batch, dtype=torch.bool).unsqueeze(1).unsqueeze(1)

                # Yield two copies of the batch and mask.
                yield reviews_tensor, sentiment_batch_tensor, padding_batch_tensor

                # reset the current batch to an empty list
                reviews = []
                sentiments = []
                max_len = 0