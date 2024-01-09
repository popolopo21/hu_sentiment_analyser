# Hungarian Sentiment Analyzer

## Overview
This project aims to implement and train a sentiment analysis model specifically for the Hungarian language, based on the architecture proposed in [this research paper](https://doi.org/10.1016/j.array.2022.100157). Initially presented at the CINTI 2023 conference, the model has been trained on a dataset of over 300,000 reviews. This README provides a comprehensive guide to the project's structure, setup, and usage.

## Model Description
The model architecture is adapted from the referenced paper with specific adjustments for the Hungarian language and to the dataset.
![model](https://github.com/popolopo21/hu_sentiment_analyser/assets/89850285/85915a97-ac86-48b0-b4f8-39dfd04b4805)

## Dataset Description

The mock dataset used for demonstration purposes is a scaled-down version of the original dataset. It includes essential preprocessing steps such as duplicate removal, NaN handling, and character count restrictions. While not as extensive as the full dataset, it serves as a representative sample for testing and development.

## Installation Instructions
First clone the repository
```
git clone 
```
Enter the directory
```
cd hu_sentiment_analyser 
```
Install dependencies
```
poetry install 
```
Run the training
```
poetry run python .\src\main.py 
```
Run mlflow to see the metrics, or go to your dagshub account.
```
mlflow run
```

## File Structure
1. **Data Ingestion**:  It downloads the dataset.
2. **Data Preprocess**: It separates the dataset into 3 parts: training, validation and test data.
3. **Prepare Base Model**: It contains the base model architecture.
4. **Training**: It trains the model on training dataset, check the config.yaml file for its output.
5. **Evaluation**: It evaluates the model and uses mlflow for storing the expreminents and models.
## Usage Instructions
Follow these steps to run each stage of the pipeline:
1. **Data Ingestion**: Run `data_ingest.py` to download and preprocess the dataset.
2. **Data Preprocess**: Execute `data_preprocess.py` to prepare the data.
3. **Prepare Base Model**: Use `model_init.py` to initialize the model architecture.
4. **Training**: Run `train_model.py` to train the model on the dataset.
5. **Evaluation**: Execute `evaluate_model.py` to generate the evaluation report.

## Configuration Details
Refer to `.env.example` for environment variable setups. Here you can setup your dagshub account, but also you can leave it empty, then the evaluation ouput will be stored in the local mlflow directory.
You can change the model's parameters and the training's settings in the `params.yaml` file

## Evaluation Metrics and Results
The model is evaluated based on accuracy, precision, and recall. A summary of these metrics, along with a confusion matrix is stored in the environment file specified or in the local mlflow directory.

## References
- Original research paper: [Link to Paper](https://doi.org/10.1016/j.array.2022.100157)
- Additional materials and resources used in this project.
