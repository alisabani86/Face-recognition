import torch
import numpy as np
import face_recognition
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from facenet_pytorch import InceptionResnetV1
import pickle, os




class FaceEmbedding:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = InceptionResnetV1(
            pretrained='vggface2').eval().to(self.device)

    def crop_faces(self, image):
        image = Image.open(image).convert('RGB')
        img_np = np.array(image)

        # Use face_recognition library for face detection
        face_locations = face_recognition.face_locations(img_np)

        if len(face_locations) == 0:
            return "No face detected"
        elif len(face_locations) > 1:
            return "More than one face detected"
        else:
            # Extract the bounding box coordinates
            top, right, bottom, left = face_locations[0]

            # Crop the face 
            face = image.crop((left, top, right, bottom))
            return face

    def extract_embedding(self, face):
        # Resize face image to match input size of FaceNet model
        face = face.resize((160, 160))
        face = F.to_tensor(face).unsqueeze(0).to(self.device)
        embedding = self.embedding_model(face)
        return embedding.detach().cpu().numpy()

    # Function to load face data from the database
    def load_face_data(self):
        if os.path.isfile('database/face_data.pickle'):
            with open('database/face_data.pickle', 'rb') as f:
                face_data = pickle.load(f)
        else:
            face_data = {}
        return face_data


    # Function to save face data to the database
    def save_face_data(self, face_data):
        with open('database/face_data.pickle', 'wb') as f:
            pickle.dump(face_data, f)

    # Function to add face embedding to the face data
    def add_face_embedding(self, face_data, name, face_embedding):
        
        if name in face_data:
            # Handle name conflicts
            counter = 1
            original_name = name
            while name in face_data:
                name = f"{original_name}{counter}"
                counter += 1

            face_data[name] = [face_embedding]
        else:
            face_data[name] = [face_embedding]

    # Function to register a face
    def register_face(self,face_embedding, name):

        face_data = self.load_face_data()
        self.add_face_embedding(face_data, name, face_embedding)
        self.save_face_data(face_data)
        print("Face registered successfully!")
