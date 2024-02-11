# Importing all the Necessary libraries
import os
import time
import tarfile
import wget

class DataInitializer():

    def __init__(self,
                 dataset_dir,
                 data_source_url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                 ):

        """
        A Utility class which contains the Helper Functions to create a dataset directory,
        Download the imdb dataset zip file which contain 25000 Training Data (12500 positive reviews and 12500 negative reviews) and 25000 Testing Data (12500 positive reviews and 12500 negative reviews),
        Extract the zip file which contain all the above data in txt files.
        
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
            prepare_dataset_folder() -> str:
                prepare the dataset folder which contains the extracted dataset files.

            check_if_file_exists(file) -> bool:
                check if the tar data file exist or not.

            check_if_dir_exists(dir) -> bool:
                check if the given directory exist or not.
        """

        self.dataset_dir = dataset_dir
        self.data_source_url = data_source_url

    def prepare_dataset_folder(self):

        """
        Function which Prepare (create, download and extract dataset contents) the Dataset Folder.

        Arguments:
            None
        Returns:
            the path of Dataset Folder which contains the extracted files.
        """

        # Create Directory if Doesn't Exist
        print("Checking if the dataset directory already exist...")
        if not self.check_if_dir_exists(self.dataset_dir):
            print("Creating a Directory for the Dataset")
            os.makedirs(self.dataset_dir)
            #os.chdir(self.dataset_dir)

        else:
            print("Not Going to Create a Directory as it Already Exist!")

        # Download the Dataset tar file using wget and place it in the created dataset directory
        print("Checking if the Dataset File Already Exist...")
        if not self.check_if_file_exists(self.dataset_dir+'/dataset.tar.gz'):
            print('Start of dataset file download...')
            wget.download(url=self.data_source_url, out=self.dataset_dir)
            print('Download complete!')

        else:
            print('Data file already exists. Not downloading again!')

        # Extract All Contents of the IMDB Dataset file into Dataset Directory
        print("Checking whether the Dataset File Already Extracted or Not...")
        if not self.check_if_dir_exists(self.dataset_dir+'/dataset'):

            print("Extracting the Dataset File as it's not Already Extracted...")
            start = time.time()

            # Open the tar archive for reading
            with tarfile.open(self.dataset_dir+'/dataset.tar.gz', 'r') as tar:
            # Extract all contents of the archive to the specified directory
                tar.extractall(path=self.dataset_dir)
            print('Extracted Successfully!')

            end = time.time()
            total_time = (end-start)/60
            print('Time Taken for extracting all files : ',total_time,'minutes')

        else:
            print('Data folder exists. Won\'t Extract again!')

        return self.dataset_dir+'/dataset'

    def check_if_file_exists(self, file):

        try:
            tarfh = tarfile.open(file)
            return True
        except FileNotFoundError:
            # print('Please make sure file: ' + file + ' is present before continuing')
            return False

    def check_if_dir_exists(self, directory):

        return(os.path.isdir(directory))
