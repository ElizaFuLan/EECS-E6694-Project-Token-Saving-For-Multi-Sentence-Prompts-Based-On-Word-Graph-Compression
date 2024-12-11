import os
import re
from typing import Dict, List, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

Graph = Dict[str, Dict[str, Union[int, float]]]
Sentence = List[str]

def parse(content: str) -> List[Sentence]:
    result = []

    content = re.sub(r'<.*?>', ' ', content)
    content = re.sub(r'[:;=8xX][-~]?[)(DPdOo/\\|*]+', ' ', content)
    content = re.sub(r'[^\w\s.,!?]', ' ', content)
    content = re.sub(r'\s+', ' ', content).strip()
    sentences = re.split(r'[.!?]', content)
    sentences = [s.strip() for s in sentences if s.strip()]

    for sentence in sentences:
        if len(re.findall(r'\b\w+\b', sentence)) == 0:
            continue

        tokens = ['<START>']
        words = re.findall(r'\b\w+\b', sentence.lower())
        tokens.extend(words)
        tokens.append('<END>')
        result.append(tokens)

    return result

def encode(sentences: List[Sentence]) -> Tuple[Graph, Dict]:
    graph = {}
    for sentence in sentences:
        prev = sentence[0]
        for current, next_ in zip(sentence[1:-1], sentence[2:]):
            if prev not in graph:
                graph[prev] = {}
            graph[prev][current] = graph[prev].get(current, 0) + 1
            prev = current
        if prev not in graph:
            graph[prev] = {}
        graph[prev][sentence[-1]] = graph[prev].get(sentence[-1], 0) + 1
    return graph, {}

def weight_graph(graph: Graph, sentences: List[Sentence]) -> Graph:
    weighted_graph = {}
    all_tokens = [' '.join(sentence) for sentence in sentences]

    if not all_tokens:
        print("Warning: No valid tokens for TF-IDF calculation.")
        return weighted_graph

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_tokens)
    tfidf_scores = {word: tfidf for word, tfidf in zip(vectorizer.get_feature_names_out(), tfidf_matrix.max(axis=0).toarray()[0])}

    for tail, heads in graph.items():
        total = sum(heads.values())
        weighted_graph[tail] = {}
        for head, count in heads.items():
            pos = pos_tag([head])[0][1]
            if pos in ['JJ', 'RB']:
                tfidf_weight = tfidf_scores.get(head, 1.0) * 3.0
            elif pos in ['NN', 'VB']:
                tfidf_weight = tfidf_scores.get(head, 1.0) * 2.0
            else:
                tfidf_weight = tfidf_scores.get(head, 1.0)

            weighted_graph[tail][head] = (1 - (count / total)) * tfidf_weight
    return weighted_graph

def remove_duplicates(sentences: List[str]) -> List[str]:
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    return unique_sentences

def filter_sentences_by_similarity(sentences: List[str], threshold: float = 0.7) -> List[str]:
    if not sentences:
        return []
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    if tfidf_matrix.shape[1] == 0:
        print("Warning: Empty vocabulary; returning original sentences.")
        return sentences
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    selected_sentences = []
    selected_indices = set()
    for i in range(len(sentences)):
        if i in selected_indices:
            continue
        selected_sentences.append(sentences[i])
        selected_indices.add(i)
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i, j] > threshold:
                selected_indices.add(j)
    return selected_sentences

def traverse(graph: Graph, max_len: int = 5) -> List[Sentence]:
    paths = []
    fringe = [(['<START>'], 0)]
    while fringe and len(paths) < max_len:
        path, cost = fringe.pop(0)
        tail = path[-1]
        if tail == '<END>' and len(path) > 3:
            paths.append((path, cost / len(path)))
            continue
        for head, weight in graph.get(tail, {}).items():
            if head not in path:
                fringe.append((path + [head], cost + weight))
    paths.sort(key=lambda x: x[1])
    return [p[0] for p in paths if p[0][-1] == '<END>']

def process_file(input_path: str, output_path: str):
    print(f"Processing file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    sentences = re.split(r'[.!?]', content.strip())
    compressed = []
    for sentence in sentences:
        if len(sentence.split()) < 3:
            continue
        tokens = parse(sentence)
        graph, _ = encode(tokens)
        weighted_graph = weight_graph(graph, tokens)
        paths = traverse(weighted_graph)
        compressed.extend([' '.join(path[1:-1]) for path in paths])

    compressed = remove_duplicates(compressed)
    compressed = filter_sentences_by_similarity(compressed)
    compressed_text = '\n'.join(compressed)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(compressed_text)

def process_dataset(input_dir: str, output_dir: str):
    for sentiment in ["neg"]:
        input_path = os.path.join(input_dir, sentiment)
        output_path = os.path.join(output_dir, sentiment)
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith('.txt'):
                    input_file = os.path.join(root, file)
                    relative_path = os.path.relpath(input_file, input_path)
                    output_file = os.path.join(output_path, relative_path)
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    process_file(input_file, output_file)

if __name__ == "__main__":
    base_dir = "aclImdb"  # IMDB dataset path
    output_base_dir = "compressed_aclImdb_clean"
    for folder in ["test"]:
        input_folder = os.path.join(base_dir, folder)
        output_folder = os.path.join(output_base_dir, folder)
        process_dataset(input_folder, output_folder)