import pandas as pd
import random

def load_file(filename):
    tmp = []
    for df in pd.read_json(filename, lines=True, chunksize=5000): 
            tmp.append(df[['subject_doc_id']])  
        
    data = pd.concat(tmp, axis=0)
    data = data.dropna(axis=0)
    data = data.drop_duplicates()
    data = data['subject_doc_id'].tolist()
    
    return data

def random_weights(data):
    docs = []
    for doc in random.sample(data,k=int(len(data)/5)):
        docs.append((doc, random.randrange(2, 100)))  
    
    return docs

def write_file(data, filename):
    with open(filename, "w") as f:
        for doc, weight in data:
            f.write(f"{doc},{weight}\n")        

doc_weights = random_weights(load_file("issuu_100k.json"))
write_file(doc_weights, "doc_weights.csv")
