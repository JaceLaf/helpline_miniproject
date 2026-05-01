from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
from sklearn.model_selection import train_test_split
import EscalationClassifier
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')

# Creates a tokenizer and encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
encoder = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
encoder.eval()

# Pulls HF emotion classification model
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Tokenizes transcript by sentences
def transcript_to_turns(text):
    return sent_tokenize(text)

# Converts sentences to embeddings
def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = encoder(**inputs)

    return outputs.last_hidden_state[0, 0].cpu()

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

def smooth(scores, k=3):
    out = []
    for i in range(len(scores)):
        window = scores[max(0, i-k):i+1]
        out.append(sum(window) / len(window))
    return out

def deltas(scores):
    return np.array([scores[i] - scores[i-1] for i in range(1, len(scores))])

def normalize_deltas(all_deltas):
    mean = np.mean(all_deltas)
    std = np.std(all_deltas) + 1e-8
    return (all_deltas - mean) / std

def label_turns(scores, z_thresh=0.75):
    scores = smooth(scores, k=3)
    all_deltas = deltas(scores)
    norm = normalize_deltas(all_deltas)

    labels = []
    state = 0

    for delta in norm:
        if delta > z_thresh:
            # Escalation
            state = 1
        elif delta <= -z_thresh:
            # De-escalation
            state = 0

        labels.append(state)

    return labels

# Determines label for transcript
def process_raw_transcript(text):
    turns = transcript_to_turns(text)

    if len(turns) < 3:
        return None
    
    # Determines escalation based on change in emotions
    scores = [emotion_score(t) for t in turns]
    labels = label_turns(scores)

    X = []
    y = []

    for i in range(1, len(turns)):
        pair = turns[i-1] + " [SEP] " + turns[i]
        X.append(embed(pair))
        y.append(labels[i-1])

    # Creates tensor objects
    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)

    return X, y

def process_with_rubric(text, rubric):
    X, y = process_raw_transcript(text)
    return None

def pipeline(df, raw=True):
    dataset = []

    # Converts transcript into escalation embeddings
    for _, row in df.iterrows():
        if raw:
            result = process_raw_transcript(row["Transcript"])
        else:
            result = process_with_rubric(row["Transcript"], row["Rubric"])
        
        if result is not None:
            dataset.append(result)

    # Splits data
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=67)

    print(len(dataset))
    print(len(train_data))
    print(len(test_data))

    # Trains and tests NN
    classifer = EscalationClassifier.run_classifier(train_data)
    EscalationClassifier.eval_classifer(classifer, test_data)