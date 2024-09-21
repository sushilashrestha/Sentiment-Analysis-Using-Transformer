# Nepali Beauty Product Sentiment Analysis Using Transformer

This project implements a sentiment analysis model for Nepali text specifically focused on beauty product reviews. It uses a transformer-based architecture to analyze sentiments in customer feedback about beauty products.

## Project Overview

The model is designed to understand and classify sentiments in Nepali language reviews of beauty products. This specialized focus allows for more accurate sentiment analysis in the beauty and cosmetics domain.

## Key Features

- Sentiment analysis of Nepali text for beauty product reviews
- Transformer-based architecture for advanced natural language processing
- Custom data preprocessing pipeline for Nepali language

## Project Structure

- `main.py`: The main script to run the entire pipeline
- `config.yaml`: Configuration file for model and training parameters
- `dataset_initializer.py`: Initializes and prepares the beauty product review dataset
- `dataset_preparer.py`: Prepares the dataset for training
- `dataset_preprocessor.py`: Preprocesses the dataset, handling beauty-specific terminology
- `batch_iterator.py`: Handles batch creation for training
- `trainer.py`: Contains the training loop and logic
- `encoder.py`: Defines the encoder model architecture
- `embedding_layers.py`: Implements custom embedding layers
- `transformer_block.py`: Defines the transformer block
- `helpers.py`: Utility functions for the project
- `predict.py`: Script for making predictions on new beauty product reviews
- `demo.py`: Demonstration script for the model
- `deploy.py`: Script for deploying the model (if applicable)

## Setup and Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/nepali-beauty-sentiment-analysis.git
   cd nepali-beauty-sentiment-analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Configure the model and training parameters in `config.yaml`.

2. Run the main script to train the model:

   ```
   python main.py
   ```

3. For predictions on new beauty product reviews, use:

   ```
   python predict.py
   ```

4. To run a demonstration:
   ```
   python demo.py
   ```

## Model Architecture

The model uses a transformer-based architecture optimized for Nepali beauty product reviews:

- Encoder with multiple transformer blocks
- Custom embedding layers (defined in `embedding_layers.py`)
- Transformer blocks (defined in `transformer_block.py`)
- Tailored to capture nuances in beauty product terminology and expressions

## Data

The project uses a dataset of Nepali beauty product reviews. The data preprocessing pipeline includes:

- Initialization (`dataset_initializer.py`)
- Preparation (`dataset_preparer.py`)
- Preprocessing (`dataset_preprocessor.py`)

These steps ensure that beauty-specific terms and expressions are properly handled.

## Training

Training is managed by `trainer.py`, which utilizes `batch_iterator.py` for efficient data handling during the training process. The model is trained to recognize sentiments specific to beauty product reviews.
