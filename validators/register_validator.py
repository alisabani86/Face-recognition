# utils/validators.py
import re


def validate_input(name, image_file):
    if not name:
        return False, "Name is required"
    if not image_file:
        return False, "Image is required"
    if not validate_name(name):
        return False, "Invalid name format"
    if not validate_image(image_file):
        return False, "Invalid image file"
    return True, None


def validate_image(image_file):
    # Perform image validation logic
    if image_file and allowed_file_extension(image_file.filename):
        return True
    else:
        return False


def validate_name(name):
    pattern = r"^[a-zA-Z0-9\s]+$"
    if re.match(pattern, name):
        return True
    else:
        return False


def allowed_file_extension(filename):
    # Perform allowed file extension validation logic
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions
