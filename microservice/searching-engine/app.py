#!/usr/bin/env python

import os
import tempfile
from flask import Flask, redirect, url_for, request, jsonify, send_file, render_template
from search import SearchingEngine
from models import Image
import io

IMAGES_DIR = os.environ.get('IMAGES_DIR') or "/Users/shengyipan/WDShares/demo/bag_test"

app = Flask(__name__)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'jpg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Path to the addtional config file that contains additional running environment related configurations
API_TOKEN = os.environ.get('REST_API_TOKEN')
# This is the path to a CSV index file
INDEX_FILE = os.environ.get('INDEX_FILE') or "/Users/shengyipan/WDShares/demo/final_result.csv"
INDEX_DICT = SearchingEngine(index_csv_file=INDEX_FILE).index

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/images/help')
def search_help():
    headers = request.headers
    auth = headers.get("X-Api-Key")
    if auth != API_TOKEN:
        return jsonify({"message": "ERROR: Unauthorized"}), 401
    return jsonify({"message": """\
Training with WML:
    URL: http://{HOST}:{PORT}/image/search
    Body: 
    1. builder_file=path_to_file
    2. (optional) extra_args={"key": "value"}
    CURL Example: 
    curl -X POST -H 'Content-Type: multipart/form-data' \
-F 'image_file=@/Users/shengyipan/Documents/demo/test.jpg' \
http://localhost:8543/image/search
"""}), 200

@app.route('/images/get')
def get_image():
    image_name = request.args.get('image_name')
    image_path = os.path.join(IMAGES_DIR, image_name)
    try:
        f = open(image_path, "rb")
        image_binary = f.read()
    except:
        return jsonify({"message": "ERROR: image not found"}), 404
    return send_file(
        io.BytesIO(image_binary),
        mimetype='image/jpeg',
        as_attachment=True,
        attachment_filename='%s' % image_name)

@app.route('/images/search', methods=['POST'])
def search():
    if request.method == 'POST':
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
                           statusCode=404), 404
        return jsonify(isError=False,
                       imageNames=similar_files,
                       message='Job completed',
                       statusCode=200), 200

@app.route('/ui')
def index():
    images = Image.sample(IMAGES_DIR)
    return render_template('demo.html', images=images)

@app.route('/ui/images/display_results')
def display_results():
    image_name = request.args.get('image_name')
    #search_image = os.path.join(IMAGES_DIR, image_name)
    similar_files = INDEX_DICT.get(image_name)
    results = [i for i in similar_files]
    if not similar_files:
        return jsonify(isError=True,
                       message='No similar images are found, double check your file name!',
                       statusCode=404), 404
    return render_template("results.html", search_image=image_name, results=results)

if __name__ == "__main__":
    app.logger.info('Listening on http://localhost:8543')
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8543)))
