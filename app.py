#!C:\handwritten_recognition_project\myenv\Scripts\flask.exe
#!C:\handwritten_recognition_project\venv\Scripts\python.exe
import os
import cv2 # type: ignore
import numpy as np # type: ignore
import requests # type: ignore
from flask import Flask, request, jsonify, send_from_directory # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Set folder for image uploads and outputs
UPLOAD_FOLDER = 'uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Horizontal Projection Method (for line segmentation)
def horizontal_projection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    hist = np.sum(thresh, axis=1)  # Sum over rows (horizontal projection)
    
    lines = []
    start = None
    for i, val in enumerate(hist):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            lines.append((start, i))
            start = None
    if start is not None:
        lines.append((start, len(hist)))  # Last line
    return lines

# Vertical Projection Method (for word segmentation)
def vertical_projection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    hist = np.sum(thresh, axis=0)  # Sum over columns (vertical projection)

    words = []
    start = None
    for i, val in enumerate(hist):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            words.append((start, i))
            start = None
    if start is not None:
        words.append((start, len(hist)))  # Last word
    return words

# Download image from URL and save to disk
def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        file_name = os.path.join(app.config['UPLOAD_FOLDER'], url.split('/')[-1])
        cv2.imwrite(file_name, image)
        return file_name
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Text Segmentation App</title>
    </head>
    <body>
        <h1>Welcome to Text Segmentation</h1>
        <form action="/upload" method="POST" enctype="multipart/form-data">
            <label for="file">Upload an Image:</label>
            <input type="file" name="file" id="file" required>
            <br><br>
            <label for="url">Or provide an Image URL:</label>
            <input type="url" name="url" id="url">
            <br><br>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files and 'url' not in request.form:
        return jsonify({'error': 'No file or URL provided'})

    # Check if the user uploaded a file
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    # Or if the user provided an image URL
    elif 'url' in request.form:
        url = request.form['url']
        file_path = download_image(url)
        if not file_path:
            return jsonify({'error': 'Failed to download the image'})

    # Read the image
    image = cv2.imread(file_path)

    # Line segmentation using horizontal projection
    lines = horizontal_projection(image)
    segmented_lines = []

    # Save segmented lines images
    for i, (start, end) in enumerate(lines):
        line_image = image[start:end, :]
        line_filename = f"{file_path.split('/')[-1].split('.')[0]}_line_{i+1}.png"
        line_path = os.path.join(app.config['UPLOAD_FOLDER'], line_filename)
        cv2.imwrite(line_path, line_image)
        segmented_lines.append(line_filename)

    # Word segmentation for each line
    segmented_words = []
    for i, (start, end) in enumerate(lines):
        line_image = image[start:end, :]
        words = vertical_projection(line_image)
        
        for j, (word_start, word_end) in enumerate(words):
            word_image = line_image[:, word_start:word_end]
            word_filename = f"{file_path.split('/')[-1].split('.')[0]}_line_{i+1}_word_{j+1}.png"
            word_path = os.path.join(app.config['UPLOAD_FOLDER'], word_filename)
            cv2.imwrite(word_path, word_image)
            segmented_words.append(word_filename)

    return jsonify({
        'message': 'File uploaded and processed successfully',
        'segmented_lines': segmented_lines,
        'segmented_words': segmented_words
    })

@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
