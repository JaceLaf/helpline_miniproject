from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
from sklearn.model_selection import train_test_split
import EscalationClassifier

nltk.download('punkt')
nltk.download('punkt_tab')

# Creates a tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
encoder = AutoModel.from_pretrained("distilbert-base-uncased")
encoder.eval()

# Pulls HF emotion classification model
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Converts sentences to embeddings
def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = encoder(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].squeeze(0)

def score_utterance(text):
    result = emotion_model(text)[0]
    return result['score']

# Converts emotion score to intensity
def emotion_score(text):
    result = emotion_model(text)[0]
    
    label = result['label']
    score = result['score']
    
    # Maps emotion to intensity value
    mapping = {
        "anger": 1.0,
        "disgust": 0.9,
        "fear": 0.8,
        "sadness": 0.7,
        "neutral": 0.3,
        "joy": 0.2,
        "surprise": 0.4
    }
    
    return mapping.get(label.lower(), 0.5) * score

# Tokenizes transcript by sentences
def transcript_to_turns(text):
    return sent_tokenize(text)

# Computes escalation change per turn
def compute_escalation_scores(turns):
    scores = [emotion_score(t) for t in turns]
    
    deltas = []
    for i in range(1, len(scores)):
        deltas.append(scores[i] - scores[i-1])
    
    return scores, deltas

# Converts escalation score to classification
def label_deltas(deltas, threshold=0.05):
    labels = []
    for d in deltas:
        if d > threshold:
            labels.append(2)
        elif d < -threshold:
            labels.append(0)
        else:
            labels.append(1)
    return labels

# Builds training pipeline
def process_transcript(text):
    turns = transcript_to_turns(text)
    
    if len(turns) < 2:
        return None
    
    scores, deltas = compute_escalation_scores(turns)
    labels = label_deltas(deltas)
    
    embeddings = [embed(t) for t in turns]
    
    X = embeddings[1:]
    y = labels
    
    return X, y

# Flattens the dataset and splits it
def split_embeddings(dataset):
    flat_X = []
    flat_y = []

    for X, y in dataset:
        for xi, yi in zip(X, y):
            flat_X.append(xi)
            flat_y.append(yi)

    X_tensor = torch.stack(flat_X)
    y_tensor = torch.tensor(flat_y)

    # Shuffles before split
    perm = torch.randperm(len(X_tensor))
    X_tensor = X_tensor[perm]
    y_tensor = y_tensor[perm]

    return train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

def pipeline(df):
    dataset = []
    for _, row in df.iterrows():
        result = process_transcript(row["Transcript"])
        
        if result is not None:
            dataset.append(result)

    # Splits data
    X_train, X_test, y_train, y_test = split_embeddings(dataset)

    # Trains and tests NN
    EscalationClassifier.run_classifier(X_train, y_train)
    EscalationClassifier.eval_classifer(X_test, y_test)

    return None