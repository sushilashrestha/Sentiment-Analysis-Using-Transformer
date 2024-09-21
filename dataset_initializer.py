import os
import time
import tarfile
import wget

class DataInitializer():

    def __init__(self,
                 dataset_dir,
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
            print("Couldn't find the dataset directory")
            exit
            #os.chdir(self.dataset_dir)

        else:
            print("Directory Exist!")
       
        return self.dataset_dir+'/dataset'

    def check_if_dir_exists(self, directory):

        return(os.path.isdir(directory))
