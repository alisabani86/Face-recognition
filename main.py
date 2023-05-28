#!/usr/bin/env python3.8
from flask import Flask
from api import (app_register,
                face_recognize
                )

app = Flask(__name__)
app.register_blueprint(app_register)
app.register_blueprint(face_recognize)



@app.route('/')
def index():
    return 'Welcome'

# run in 0.0.0.0 to make accesable for public
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
