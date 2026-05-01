import kagglehub
import os
import pandas as pd
import model
from openai import OpenAI

# Uses GPT-4.1-mini to generate a rubric for each transcript
client = OpenAI()

def extract_rubric(transcript, rubric):
    prompt = rubric + "\n\nTRANSCRIPT:\n" + transcript
    return call_llm(prompt)

def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a precise information extraction system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

# Loads in call center files into a df
path = kagglehub.dataset_download("oleksiymaliovanyy/call-center-transcripts-dataset")
file = os.listdir(path)
file = [f for f in file if f.endswith(".csv") and "description" not in f.lower()]
df = pd.read_csv(os.path.join(path, file[0]))

# Reads in rubric
rubric_ptr = open("/home/jason-lafita/helpline_miniproject/rubric.txt", "r")
rubric_prompt = rubric_ptr.read()
rubric_ptr.close()

rubs = []
for t in df["Transcript"]:
    r = extract_rubric(t, rubric_prompt)
    rubs.append(r)

df["Rubric"] = rubs

# Runs model on csv file with rubric
model.pipeline(df, False)