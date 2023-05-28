from utils import (error_handlers as err,
                   consts)
from validators import register_validator
from flask import Flask, request, jsonify, Blueprint
from face.embedding import FaceEmbedding
from face.recognition import recognize_face
import os


face_embedding = FaceEmbedding()
face_recognize = Blueprint('face_recognize', __name__)


@face_recognize.route('/api/recognize', methods=['POST'])
def recognize():
    # Get the input image from the request
    image = request.files.get('image')

    #Validation 
    validation = register_validator.validate_image(image)
    if validation != True:
        return jsonify({'error': "invalid Image"}), 400
    

    #crop and embedd image
    cropped_face = face_embedding.crop_faces(image)
    image_embedding = face_embedding.extract_embedding(cropped_face)

    # Perform face recognition
    similarity_threshold = consts.SIMILARITY_THRESHOLD
    name = recognize_face(similarity_threshold, image_embedding)

    if name:
        # Return the recognized name if a match is found
        return jsonify({'message': 'Match found!', 'name': name})
    else:
        # Return a message indicating no match is found
        return jsonify({'message': 'No match found.'})