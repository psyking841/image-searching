#!/usr/bin/env python

import os
import tempfile
from flask import Flask, request, jsonify
from search import SearchingEngine

app = Flask(__name__)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'jpg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to the addtional config file that contains additional running environment related configurations
API_TOKEN = os.environ.get('REST_API_TOKEN')
# This is the path to a CSV index file
INDEX_FILE = os.environ.get('INDEX_FILE')
INDEX_DICT = SearchingEngine(index_csv_file=INDEX_FILE).index

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/search', method=['POST'])
def search():
    request.method == 'POST':
        headers = request.headers
        auth = headers.get("X-Api-Key")
        if auth != API_TOKEN:
            return jsonify({"message": "ERROR: Unauthorized"}), 401
        if 'image_file' not in request.files:
            return jsonify(isError=True,
                           message='No image file is provided',
                           statusCode=200), 200
        file = request.files['image_file']
        if (not file) or (not allowed_file(file.filename)):
            return jsonify(isError=True,
                           message='Image file is not valid',
                           statusCode=200), 200
        # Look up the table for now; TODO: compute the similar image.
        similar_files = INDEX_DICT.get(file.filename)
        if not similar_files:
            return jsonify(isError=True,
                           message='No similar images are found, double check your file name!',
                           statusCode=200), 200
        return jsonify(isError=False,
                       imageNames=similar_files,
                       message='Job completed',
                       statusCode=200), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8543)))
