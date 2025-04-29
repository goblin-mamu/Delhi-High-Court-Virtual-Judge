from flask import Flask, render_template, request, jsonify
import os
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """Renders the root website."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads and returns large data."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        time.sleep(3)
        large_data = {
            'result': 'File uploaded and processed successfully!',
            'data_points': [f'Data Point {i}' for i in range(1000)]  # Example large data
        }
        return jsonify(large_data)

if __name__ == '__main__':
    app.run(debug=True)