

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
    app.run(host='0.0.0.0', port=5000)