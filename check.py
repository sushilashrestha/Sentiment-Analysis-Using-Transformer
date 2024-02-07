import os
import shutil

class DataInitializer():
    def __init__(self, dataset_dir, data_source_file):
        self.dataset_dir = dataset_dir
        self.data_source_file = data_source_file

    def prepare_dataset_folder(self):
        # Create directory if it doesn't exist
        print("Checking if the dataset directory already exists...")
        if not os.path.exists(self.dataset_dir):
            print("Creating a directory for the dataset")
            os.makedirs(self.dataset_dir)

        # Check if the dataset file exists
        print("Checking if the dataset file exists...")
        if os.path.isfile(self.data_source_file):
            # Move the dataset file to the dataset directory
            shutil.copy(self.data_source_file, self.dataset_dir)
            print("Dataset file copied successfully!")
        else:
            print("Dataset file not found!")

        return self.dataset_dir

dataset_dir = "C:\\Users\\admin\\Desktop\\reviews"
data_source_file = "C:\\Users\\admin\\Desktop\\reviews\\translated\\cocooncenter_translated.csv"

data_initializer = DataInitializer(dataset_dir, data_source_file)
dataset_folder = data_initializer.prepare_dataset_folder()
print("Dataset folder:", dataset_folder)