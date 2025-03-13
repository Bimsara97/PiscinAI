import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from PIL import Image
import io
import base64
import numpy as np
from utils.model_utils import load_disease_model, predict_disease

app = Flask(__name__)
app.secret_key = 'fish_analysis_app_secret_key'

# Constants
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/fish_disease_model.keras'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable for model
disease_model = None

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user does not select file
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Read and save the image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Get predictions
        result = predict_disease(disease_model, image)
        
        # Save file for display
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'latest_upload.jpg')
        image.save(filename)
        
        return render_template('result.html', 
                               result=result, 
                               image_path=filename,
                               mode='upload')
    
    flash('Invalid file type. Please upload an image file (png, jpg, jpeg).')
    return redirect(url_for('index'))

@app.route('/capture', methods=['POST'])
def capture():
    """Handle camera capture and analysis"""
    try:
        # Get the image data from the POST request
        image_data = request.form['image_data']
        
        # Remove the 'data:image/jpeg;base64,' prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get predictions
        result = predict_disease(disease_model, image)
        
        # Save file for display
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'latest_capture.jpg')
        image.save(filename)
        
        return render_template('result.html', 
                               result=result, 
                               image_path=filename,
                               mode='capture')
    except Exception as e:
        print(f"Error processing captured image: {e}")
        flash('Error processing captured image. Please try again.')
        return redirect(url_for('index'))

# API endpoint for mobile apps if needed
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analyzing fish images"""
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Read the image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Get predictions
        result = predict_disease(disease_model, image)
        
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Load models when app starts
    disease_model = load_disease_model(MODEL_PATH)
    
    # Run app
    app.run(debug=True, host='0.0.0.0')