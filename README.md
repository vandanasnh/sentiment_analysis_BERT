## Fine-Tuning BERT for Sentiment Analysis

The data used consists of 3,000 customer reviews labeled as positive (1) or negative (0). The data is split into a training set of 2,400 samples and a testing set of 600 samples.

![Bert Model Flow][link here](https://github.com/vandanasnh/sentiment_analysis_BERT/blob/main/screenshot.png)

**Data Processing**

1. **Loading Data:** The script reads the data from text files containing labeled sentences. It then converts the data into a TensorFlow dataset.

2. **Preprocessing:** The text data is preprocessed using a pre-trained BERT preprocessor model (`bert_preprocess_model`). This converts the text into numerical representations that the BERT encoder can understand.

3. **Encoding:** The preprocessed data is then fed into the BERT encoder model (`bert_model`). The encoder generates a vector representation for each sentence that captures its meaning.

**Building the Model**

1. **Classifier Pipeline:** A function (`classifier_pipeline`) is defined to create a neural network classifier on top of the pre-trained BERT model. This function takes the number of intermediate layers and the number of units in each layer as arguments.

2. **Loss Function and Optimizer:** The binary cross-entropy loss function is used for binary classification (positive or negative sentiment). The AdamW optimizer with a learning rate scheduler is used for training.

3. **Model Compilation:** The classifier model is compiled with the chosen loss function, optimizer, and metrics (accuracy).

**Training and Testing**

1. **Training:** The model is trained on the training data for a specified number of epochs. The validation accuracy is monitored during training to prevent overfitting.

2. **Evaluation:** After training, the model is evaluated on the testing data to assess its performance. The evaluation metrics include loss and accuracy.

3. **Visualization:** A graph is plotted to visualize the training and validation accuracy over the training epochs.

**Saving and Loading the Model**

1. **Saving:** The trained model is saved to a specific directory for future use.

2. **Loading:** The saved model can be loaded back using TensorFlow's `load_model` function.


**Additional Notes**

* The script includes comments throughout the code to explain each step.
* The script assumes the `tensorflow`, `tensorflow_hub`, `tensorflow_text`, and `official.nlp` libraries are installed.
* The script uses a small pre-trained BERT model (`bert_en_uncased_L-4_H-512_A-8`) for demonstration purposes. You can experiment with different BERT models based on your needs.


I hope this readme provides a clear understanding of the fine-tuning process for sentiment analysis using BERT!
