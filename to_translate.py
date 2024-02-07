import pandas as pd
import os
import time
from httpcore._exceptions import ReadTimeout
from googletrans import Translator
from nepali_unicode_converter.convert import Converter

converter = Converter() 
translator = Translator()

inputFileName: str = "part4.csv"
outputFileName: str = f"translated_{inputFileName}"
reviewRowName: str = "review_text"
nepaliReviewRowName: str = "nepali_review"

def translate_with_retry(text):
    while True:
        try:
            result = translator.translate(text,src= 'en' ,dest="ne")
            return result.text
        except TimeoutError:
            print("Timeout error. Retrying...")
            time.sleep(2)  # Add a delay before retrying
            continue   
        except TypeError:
            print("Type Error has been noticed.")
            print(f"The function is returning None for input of {text}")
            break
        except ReadTimeout:
            print(f"Request timed out. Retrying...")
            time.sleep(2)  # Add a delay before retrying
            continue
        except Exception as e:
            print(f"Error has been noticed. {e}")
            break


# ---------------- Read the CSV file ----------------
try:
    df = pd.read_csv(inputFileName)
except FileNotFoundError:
    print("File not found.")
    exit()
except pd.errors.EmptyDataError:
    print("File is empty.")
    exit()
else:
    print("CSV file read successfully.")

    df[reviewRowName] = df[reviewRowName].astype(str)
    print("Data type of 'review' column converted to string.")

    if nepaliReviewRowName not in df.columns:
        df[nepaliReviewRowName] = None
        print("Added 'nepali_review' column to the dataframe.")


# ---------------- Checking Output File ----------------
if (not os.path.isfile(outputFileName)) or (os.stat(outputFileName).st_size == 0):
    df.head(0).to_csv(outputFileName, mode="w", header=True, index=False)


# ---------------- Resume Functionality ----------------
try:
    with open("last_translated_row.txt", "r") as f:
        start_index = int(f.read()) + 1

except FileNotFoundError:
    # create a file
    with open("last_translated_row.txt", "w") as f:
        f.write("0")
    start_index = 0

except ValueError:
    start_index = 0


# ---------------- Translate the reviews ----------------
for index, row in df.loc[start_index:].iterrows():

    print(f"Translating row {index}...")
    nepali_review = translate_with_retry(row[reviewRowName])
    df.at[index, nepaliReviewRowName] = nepali_review
    # Write the translated row to the CSV file
    row[nepaliReviewRowName] = nepali_review
    row.to_frame().T.to_csv(outputFileName, mode="a", header=False, index=False)
    # Update the last translated row index
    with open("last_translated_row.txt", "w") as f:
        f.write(str(index))

print("Translation process completed.")

# delete the last translated row file
os.remove("last_translated_row.txt")
