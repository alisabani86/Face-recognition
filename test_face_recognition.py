import os
from validators import register_validator
from face.embedding import FaceEmbedding
from face.recognition import recognize_face as face_recognition
from sklearn.metrics import classification_report
from PIL import Image
from utils import consts

# Initialize the FaceEmbedding class
face_embedding = FaceEmbedding()

# Register a face from an image
def register_face(image, name):
    
    # Process the image (crop and embed)
    cropped_face = face_embedding.crop_faces(image)
    embedding = face_embedding.extract_embedding(cropped_face)

    # Save the embedded face to the pickle file
    face_embedding.register_face(embedding, name)

    print(f"Registered face: {name} from image: {image_path}")

# List of image names and corresponding names
image_names = [
    ('image1.jpg', 'John'),
    ('image2.jpg', 'David'),
    ('image3.jpg', 'Michael'),
    ('image4.jpg', 'James'),
    ('image5.jpg', 'William'),
    ('image6.jpg', 'Benjamin'),
    ('image7.jpg', 'Daniel'),
    ('image8.jpg', 'Matthew'),
    ('image9.jpg', 'Ethan'),
    ('image10.jpg', 'Joseph'),
    ('image11.jpg', 'Andrew'),
    ('image12.jpg', 'Ryan'),
    ('image13.jpg', 'Joshua'),
    ('image14.jpg', 'Christopher'),
    ('image15.jpg', 'Nathan'),
    ('image16.jpg', 'Samuel'),
    ('image17.jpg', 'Kevin'),
    ('image18.jpg', 'Brandon'),
    ('image19.jpg', 'Justin'),
    ('image20.jpg', 'Brian'),
    ('image21.jpg', 'Jonathan'),
    ('image22.jpg', 'Jason'),
    ('image23.jpg', 'Aaron'),
    ('image24.jpg', 'Eric'),
    ('image25.jpg', 'Tyler'),
]

# Register faces from the list
for image_name, name in image_names:
    # Construct the image path based on the image name
    image_path = os.path.join('test/dataset', image_name)

    # Register the face
    register_face(image_path, name)

predicted_labels = []
true_labels = []

# Test Face Recognition
for image_name, name in image_names:
    # Construct the image path based on the image name
    image_path = os.path.join('test/dataset', image_name)
 
   
    # Crop and embed the image
    cropped_face = face_embedding.crop_faces(image_path)
    image_embedding = face_embedding.extract_embedding(cropped_face)
    # Perform face recognition
    similarity_threshold = consts.SIMILARITY_THRESHOLD
    predicted_label = face_recognition(similarity_threshold, image_embedding)
    # Store the predicted and true labels
    predicted_labels.append(predicted_label)
    true_labels.append(name)

# Generate classification report
report = classification_report(true_labels, predicted_labels)
# Save the report to a text file
with open('classification_report.txt', 'w') as f:
    f.write(report)
# Print the report
print(report)
