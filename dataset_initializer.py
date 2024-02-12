import os

class DataInitializer():

    def __init__(self,
                 dataset_dir,
                 ):

        """
        A Utility class which contains the Helper Functions to check if the dataset directory and the dataset file already exist.
        
        We are using IMDB Movie Reviews Dataset for binary sentiment classification that provides a set of 25,000 highly polar reviews for training,
        And 25,000 for testing (each set contains an equal number of positive and negative examples).

        Dataset folder structure is as follows:

        dataset/
        ├── test/
        │     ├── positive/
        │     ├── negative/
        ├── train/
              ├── positive/
              └── negative/

        Args:
            dataset_dir : Directory to place the Dataset Folder
            data_source_url : Url of the Dataset file [Optional]
        Methods:
            check_if_dir_exists(dir) -> bool:
                check if the given directory exist or not.

            check_if_file_exists(file) -> bool:
                check if the tar data file exist or not.
        """

        self.dataset_dir = dataset_dir

    def check_if_dir_exists(self, directory):

        return(os.path.isdir(directory))

    def check_if_file_exists(self, file):

        return os.path.isfile(file)

    def prepare_dataset_folder(self):

        """
        Function which Checks if the Dataset Folder and the Dataset File Already Exist.

        Arguments:
            None
        Returns:
            a tuple of two boolean values indicating the existence of the dataset directory and the dataset file respectively.
        """

        # Check if the Directory Exists
        print("Checking if the dataset directory already exist...")
        dir_exists = self.check_if_dir_exists(self.dataset_dir)

        # Check if the Dataset File Exists
        print("Checking if the Dataset File Already Exist...")
        file_exists = self.check_if_file_exists(self.dataset_dir+'/dataset')

        return dir_exists, file_exists