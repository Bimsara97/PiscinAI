import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Constants
IMAGE_SIZE = 256

# Disease class names
DISEASE_CLASSES = [
    'Bacterial Red disease',
    'Bacterial diseases - Aeromoniasis',
    'Bacterial gill disease',
    'Fungal diseases Saprolegniasis',
    'Healthy Fish',
    'Parasitic diseases',
    'Viral diseases White tail disease'
]

# Treatment recommendations based on disease
TREATMENT_RECOMMENDATIONS = {
    'Bacterial Red disease': [
        "Apply antibacterial medication as prescribed by an aquatic veterinarian",
        "Raise water temperature to 80-82°F (26-28°C) for freshwater species",
        "Ensure proper water quality with regular changes (25-30% weekly)",
        "Add aquarium salt (1 tablespoon per 5 gallons) for freshwater tanks",
        "Isolate affected fish to prevent spread"
    ],
    'Bacterial diseases - Aeromoniasis': [
        "Use broad-spectrum antibiotics like Kanamycin or Tetracycline as directed",
        "Improve water quality with frequent water changes (30% every 3 days)",
        "Reduce organic waste in tank/pond",
        "Add aquarium salt (1-2 tablespoons per 10 gallons) for freshwater species",
        "Increase aeration and ensure proper filtration"
    ],
    'Bacterial gill disease': [
        "Treat with antibiotics specific for gill disease (consult a fish veterinarian)",
        "Improve oxygen levels with additional aeration",
        "Perform frequent water changes (20-30% every other day)",
        "Add aquarium salt (1 tablespoon per 5 gallons) for freshwater species",
        "Clean gravel/substrate to reduce bacterial load"
    ],
    'Fungal diseases Saprolegniasis': [
        "Apply antifungal treatments like Methylene Blue or Malachite Green",
        "Perform daily water changes (25%) during treatment",
        "Increase water temperature slightly (2-3°F above normal)",
        "Remove visible fungal growth carefully with a cotton swab",
        "Ensure good water circulation and filtration"
    ],
    'Healthy Fish': [
        "Continue regular maintenance and feeding schedule",
        "Maintain consistent water quality parameters",
        "Perform routine water changes (20-25% weekly)",
        "Ensure balanced nutrition with variety in diet",
        "Monitor for any behavioral or physical changes"
    ],
    'Parasitic diseases': [
        "Apply appropriate antiparasitic medication based on parasite type",
        "Raise temperature gradually to 82-86°F (28-30°C) for some parasites",
        "Perform thorough gravel vacuuming to remove parasite eggs/cysts",
        "Change 30-40% of water every 2-3 days during treatment",
        "Consider quarantine treatment tank for severe infestations"
    ],
    'Viral diseases White tail disease': [
        "Isolate affected fish immediately to prevent spread",
        "Carefully maintain optimal water conditions",
        "Support immune system with high-quality food and vitamin supplements",
        "Consider euthanasia for severely affected fish to prevent spread",
        "Disinfect equipment and nets used with affected fish"
    ]
}

def load_disease_model(model_path):
    """
    Load the disease classification model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded TensorFlow model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create a placeholder model for development if model file doesn't exist
        print("Using placeholder model for development")
        # Create a simple placeholder model for testing
        inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(len(DISEASE_CLASSES), activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

def preprocess_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """
    Preprocess an image for model input
    
    Args:
        image: PIL Image object
        target_size: Tuple of (width, height) for resizing
        
    Returns:
        Preprocessed image array ready for model input
    """
    # Convert to RGB if image is in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and expand dimensions for batch
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Normalize
    image_array = image_array / 255.0
    
    return image_array

def predict_disease(model, image):
    """
    Predict disease from image
    
    Args:
        model: Loaded TensorFlow model
        image: PIL Image object
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index] * 100)
        predicted_class = DISEASE_CLASSES[predicted_class_index]
        
        # Return results
        return {
            'disease': predicted_class,
            'confidence': confidence,
            'treatments': TREATMENT_RECOMMENDATIONS.get(predicted_class, ["No specific treatment found."])
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {
            'disease': 'Error in prediction',
            'confidence': 0,
            'treatments': ["Error occurred during analysis. Please try again."]
        }