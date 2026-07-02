# The Visual Storyteller: Image Captioning
Final project for the Deep Learning course. This project builds an image captioning model that takes an image as input and creates a sentence describing what it sees.
## Project Overview
Image captioning combines computer vision and natural language proccessing(NLP). The model first understands what is in an image and then writes a sentence about it. My project uses a CNN encoder to extract features from an image and an attention-based LSTM decoder to generate the caption one word at a time.
Dataset: 8,000 images, each with 5 captions.
# Model Design
# Architecture
```text
Image (224x224x3)
      │
EfficientNet-B0 (pretrained and frozen)
      │
Feature Map (7x7x1280)
      │
Linear Layer + Dropout
      │
Attention + LSTM Decoder
      │
Generated Caption
```
# Encoder
The encoder uses a pretrained EfficientNet-B0 model. Instead of producing one feature vector it outputs a 7×7 feature map (49 regions of the image). These regions allow the attention mechanism to focus on different parts of the image while generating the caption. The encoder stays frozen during training which makes training faster and helps prevent overfitting.
# Decoder
The decoder is an LSTM with a Bahdanau Attention mechanism. For every new word it generates the attention layer decides which part of the image is the most important. This helps model create more accurate captions.
# Training
The model is trained using:
- Cross-Entropy Loss
-  Teacher Forcing
-  Label Smoothing
-  Gradient Clipping
Only the decoder is trained, the encoder remains frozen.
# Caption Generation
During inference captions are generated using Beam Search(beam width = 3) with length normalization. 
# Main Hyperparameters
- Word embedding size: 300
- LSTM hidden size: 600
- Vocabulary minimum frequency: 5
- Maximum caption length: 30 words
- Dataset split: 80% training, 10% validation, 10% testing
# Results
Metric
- Best Validation Loss:    2.9619
- Corpus BLEU Score:       0.1736
- Average Sentence BLEU:   0.0831
- Average Inference Time:  0.734 sec/image
## Repository Structure
```text
image-captioning-project/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── data_and_training.ipynb
│   └── inference.ipynb
├── models/
│   ├── best_model.pt
│   ├── test_data.json
│   └── tokenizer.pkl
└── data/
    ├── Images/
    └── captions.txt
```
# Notebook Description
- data_and_training.ipynb – Loads the dataset, builds the vocabulary, defines the model, trains it and saves the trained model and other files.
- inference.ipynb – Loads the trained model, generates captions for new images, calculates BLEU scores and shows successful and unsuccessful examples.
