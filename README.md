# Facebook Comments Intent Classification

## Description

The Facebook Comments Intent Classification project is aimed at building a machine learning model to classify the intent behind Facebook comments. The goal is to automatically determine whether a comment expresses positive sentiment, negative sentiment, or a neutral sentiment.

The project utilizes natural language processing techniques and machine learning algorithms to train a classifier on a labeled dataset of Facebook comments. The trained model can then be used to predict the sentiment intent of new, unseen comments.

## Installation

To run the project, make sure you have the following installed:

- Python (version 3.7 or above)
- The required Python libraries specified in the `requirements.txt` file


## Dataset
The dataset used for training and evaluation is available in the data directory. It consists of a CSV file containing labeled Facebook comments, where each comment is associated with a sentiment label (positive, negative, or neutral). The dataset is split into training and test sets.

## Usage
Ensure that the dataset is available in the data directory or modify the file path in the code accordingly.

Run the train.py script to train the sentiment classification model:

bash
Copy code
python train.py
After training, you can use the trained model to predict the sentiment intent of new comments. Run the predict.py script and provide a comment as input:
bash
Copy code
python predict.py --comment "This is a great post!"
The script will output the predicted sentiment intent for the given comment.

## Customization
To modify the model architecture or hyperparameters, you can edit the code in the train.py script.

Feel free to experiment with different feature extraction techniques, such as using word embeddings or applying other text preprocessing methods, to improve the model's performance.

## Contributing
Contributions to the Facebook Comments Intent Classification project are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
