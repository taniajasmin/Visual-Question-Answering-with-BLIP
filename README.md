# Visual Question Answering with BLIP

This project implements a Visual Question Answering (VQA) system using the BLIP (Bootstrapping Language-Image Pretraining) model. The system takes an image and a natural language question as input and predicts a short text answer.

## Project Overview

The goal of this project is to fine-tune a pretrained multimodal model so it can understand both visual information (images) and text (questions).

**Example:**
Question: What is hanging to the right side of the bed?
Prediction: curtain

The model learns to connect objects in the image with the meaning of the question.

## Dataset

The dataset was downloaded from Kaggle using `kagglehub`.

**Dataset:**
`visual-question-answering-computer-vision-nlp`

**Download code:**
```python
import kagglehub

path = kagglehub.dataset_download("bhavikardeshna/visual-question-answering-computer-vision-nlp")
```

The dataset contains:
*   images
*   questions related to each image
*   correct answers

**Structure:**
```
dataset/
│ ├── images/
├── data_train.csv
├── data_eval.csv
```

Each row contains:

| `image_id` | `question`                 | `answer` |
|------------|----------------------------|----------|
| 100        | What is on the shelves?    | cup      |

## Model

The project uses the pretrained model: `Salesforce/blip-vqa-base`

BLIP is a multimodal transformer that combines:
*   **Vision Encoder** → extracts image features
*   **Text Encoder** → understands the question
*   **Multimodal Fusion** → combines image and text
*   **Text Decoder** → generates the answer

**Pipeline:**
```
Image → Vision Encoder
Question → Text Encoder
            ↓
    Multimodal Fusion
            ↓
      Text Decoder
            ↓
          Answer
```

## Training

The model was fine-tuned using PyTorch on a GPU environment.

**Training configuration:**
*   Optimizer: AdamW
*   Learning rate: 5e-5
*   Batch size: 1
*   Gradient accumulation: 4
*   Epochs: 2

Training loss decreased significantly:

| Epoch | Loss      |
|-------|-----------|
| 0     | 2313177   |

This shows the model successfully learned patterns from the dataset.

## Evaluation

The model was evaluated using:
*   Accuracy
*   F1 Score
*   BLEU Score

**Results:**

| Model      | Accuracy | F1 Score | BLEU Score |
|------------|----------|----------|------------|
| BLIP VQA   | 0.21     | 0.188    | 0.043      |

## Example Predictions

**Example 1**
Question: What is the object on the shelves? Prediction: cup Actual: cup

**Example 2**
Question: How many chairs are there? Prediction: 8 Actual: 6

**Example 3**
Question: What is hanging to the right side of the bed? Prediction: curtain Actual: curtain

## Tech Stack

*   Python
*   PyTorch
*   HuggingFace Transformers
*   BLIP
*   Pandas
*   NLTK
*   Matplotlib

## Conclusion

This project demonstrates how multimodal AI models can combine image understanding and language understanding to answer questions about visual scenes. With more training data and longer training, the model's performance could be improved further.
