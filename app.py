# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# import os
# import json
# from werkzeug.serving import run_simple
#
# from Model_ml import JobRecommender
#
# app = Flask(__name__)
#
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400
#
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected for uploading'}), 400
#
#     if file:
#         filename = secure_filename(file.filename)
#         file.save(filename)  # replace with your actual path
#         recommender = JobRecommender()
#         recommendations = recommender.process_cv(filename)  # replace with your actual path
#
#         return jsonify(recommendations)
# if __name__ == '__main__':
#     run_simple('localhost', 5000, app)



from flask import Flask, request, jsonify
from Model_ml import JobRecommender

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'url' not in request.json:
        return jsonify({'error': 'No url in the request'}), 400

    url = request.json['url']
    if url == '':
        return jsonify({'error': 'No url provided'}), 400

    recommender = JobRecommender()
    recommendations = recommender.process_cv(url)

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='localhost', port=5000)