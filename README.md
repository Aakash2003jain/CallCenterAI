# Call Center Audio Multi-Class Classification

This repository contains the code for a multi-class classification model designed to categorize call center audio data into different classes. The project leverages state-of-the-art deep learning techniques using TensorFlow, PyTorch, and the Transformers library.

## Description

In this project, we develop a machine learning model capable of classifying call center audio recordings into predefined categories. The model is built using a BERT-based architecture for handling textual data extracted from audio, and it is trained and fine-tuned for optimal performance.

## Installation

To set up the environment and install the required dependencies, please follow the steps below:

```bash
# Clone the repository
git clone https://github.com/your-username/Call_Center_Audio_Multi_Class_Classification.git

# Navigate to the project directory
cd Call_Center_Audio_Multi_Class_Classification

# Install the required Python packages
pip install -r requirements.txt

```
## Usage

### Google Colab

To run the notebook on Google Colab, follow these steps:

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `Call_Center_Audio_Multi_Class_Classification.ipynb` notebook to Colab.
3. Follow the instructions in the notebook to run each cell.

## Dataset
The dataset used in this project consists of call center audio recordings labeled into multiple categories. The audio data is preprocessed and converted into text using an automatic speech recognition (ASR) system.

## Model Architecture
The model is based on the BERT architecture for sequence classification. Key components include:

1. BERT Pretrained Model: Used to generate embeddings from the input text.
2. Classification Layer: A fully connected layer that maps the BERT embeddings to output classes.

## Training
The model is trained using the following setup:

1. Optimizer: AdamW with learning rate scheduling.
2. Loss Function: Cross-entropy loss for multi-class classification.
3. Batch Size: Configurable depending on available hardware.
4. Epochs: The model is trained for a specified number of epochs, with early stopping based on validation loss.

## Evaluation
The model is evaluated using standard classification metrics such as accuracy, precision, recall, and F1-score. The notebook includes detailed analysis and visualizations of the results.

## Results
Detailed results, including model accuracy and confusion matrix, are provided in the notebook. The final model achieves competitive performance on the test set.
