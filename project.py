import kagglehub
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import torch
import os
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

# Creates a tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
encoder = AutoModel.from_pretrained("distilbert-base-uncased")
encoder.eval()

# Converts sentences to embeddings
def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = encoder(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].squeeze(0)

# Pulls HF emotion classification model
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
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

# Loads in call center files into a df
path = kagglehub.dataset_download("oleksiymaliovanyy/call-center-transcripts-dataset")
file = os.listdir(path)
file = [f for f in file if f.endswith(".csv") and "description" not in f.lower()]
df = pd.read_csv(os.path.join(path, file[0]))

# Tokenizes transcript by sentences
def transcript_to_turns(text):
    return sent_tokenize(text)

turns = transcript_to_turns(df.iloc[0]["Transcript"])

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

# Applies model to dataset
dataset = []
for _, row in df.iterrows():
    result = process_transcript(row["Transcript"])
    
    if result is not None:
        dataset.append(result)