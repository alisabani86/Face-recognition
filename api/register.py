from validators import register_validator
from flask import Flask, request, jsonify, Blueprint
from face.embedding import FaceEmbedding

#init class face embedding
face_embedding = FaceEmbedding()

#register app with blueprint
app_register = Blueprint('register', __name__)




@app_register.route('/api/register', methods=['POST'])
def register():
    # Get image and name from request
    image = request.files.get('image')
    name = request.form.get('name')
    # Validate the inputs
    is_valid, error_message = register_validator.validate_input(name, image)

    if not is_valid:
        # Return the error message if validation fails
        return jsonify({'error': error_message}), 400
    
    #process image crop and embedd
    cropped_face = face_embedding.crop_faces(image)
    embedding = face_embedding.extract_embedding(cropped_face)

    #if success then save embedded file into pickle
    face_embedding.register_face(embedding, name)

    # Return a success response
    return jsonify({'message': 'Registration successfull'})
