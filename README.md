# Fish Analyzer App

A web application for identifying fish species, gender, diseases, and suggesting treatments using deep learning models.

## Features

- **Fish Species Identification**: Automatically recognize common freshwater fish species
- **Gender Classification**: Determine the gender of the fish
- **Disease Detection**: Identify various fish diseases and health conditions
- **Treatment Recommendations**: Get specific treatment suggestions for detected diseases
- **Camera Integration**: Take photos directly using your device's camera
- **Image Upload**: Upload existing fish photos for analysis

## Technologies Used

- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Computer Vision**: OpenCV
- **Image Processing**: PIL (Python Imaging Library)

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Webcam (for photo capture feature)

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fish-analyzer.git
   cd fish-analyzer
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Place your trained models in the `models` directory.
   - The main disease classification model should be named `fish_disease_model.keras`
   - If you have separate models for species or gender classification, place them in the same directory

5. Run the application:
   ```
   python app.py
   ```

6. Open a web browser and navigate to `http://127.0.0.1:5000/`

## Usage

1. **Upload an Image**:
   - Click on the "Upload Image" tab
   - Select a clear image of your fish
   - Click "Analyze Image"

2. **Take a Photo**:
   - Click on the "Take Photo" tab
   - Grant camera permission when prompted
   - Position your fish in front of the camera
   - Click "Take Photo"
   - Verify the captured image and click "Analyze Captured Image"

3. **View Results**:
   - See the identified fish species, gender, and any detected diseases
   - View the confidence level of the predictions
   - Read recommended treatments if a disease is detected
   - Learn more about the detected condition

## Model Training

The application uses deep learning models trained on a dataset of freshwater fish images. The models are based on the ResNet50 architecture, fine-tuned for specific tasks:

- The disease identification model was trained on approximately 2,400 images across 7 disease categories
- The models achieve over 97% accuracy on test data

## Customization

You can adapt this application to your specific needs:

- Replace the models with your own trained models
- Add or modify disease categories in `model_utils.py`
- Customize treatment recommendations in `model_utils.py`
- Update the UI by modifying the templates in the `templates` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset sourced from [Kaggle: Freshwater Fish Disease Aquaculture in South Asia](https://www.kaggle.com/datasets/subirbiswas19/freshwaterfish-disease-aquaculture-in-south-asia/data)
- TensorFlow and Keras for machine learning frameworks
- Flask for the web framework
- Bootstrap for the frontend styling