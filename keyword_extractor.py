from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_distilbert_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model

def extract_keywords(text, top_k=5):
    tokenizer, model = load_distilbert_model()
    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**tokens)
    token_embeddings = outputs.last_hidden_state.squeeze(0).detach().numpy()
    input_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze())

    sentence_embedding = np.mean(token_embeddings, axis=0)
    scores = [cosine_similarity([sentence_embedding], [emb])[0][0] for emb in token_embeddings]

    keywords = [(input_tokens[i], scores[i]) for i in range(len(input_tokens)) if input_tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)

    return [kw[0] for kw in keywords[:top_k]]