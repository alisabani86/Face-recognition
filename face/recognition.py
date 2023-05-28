import cv2
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load face embeddings from the database
def load_face_embeddings():
    with open('database/face_data.pickle', 'rb') as f:
        face_data = pickle.load(f)
    return face_data



# Compare the input image with the embeddings in the database
def compare_face_embeddings(input_embedding, face_data):
    similarities = {}
    for name, embeddings in face_data.items():
        input_embedding = input_embedding.reshape(input_embedding.shape[0], -1)
        embeddings = np.asarray(embeddings)
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        # Calculate cosine similarity between input embedding and embeddings in the database
        similarity = np.mean(cosine_similarity(input_embedding.reshape(1, -1), embeddings))
        similarities[name] = similarity
    return similarities

# Find the best match based on the highest similarity
def find_best_match(similarities, threshold):
    best_match = max(similarities, key=similarities.get)
    similarity_score = similarities[best_match]
    return best_match if similarity_score >= threshold else None

# Perform face recognition
def recognize_face(similarity_threshold, input_embedding):

    face_data = load_face_embeddings()
    similarities = compare_face_embeddings(input_embedding, face_data)
    best_match = find_best_match(similarities, similarity_threshold)
    return best_match

