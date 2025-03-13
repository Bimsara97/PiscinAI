import os
import tensorflow as tf
import argparse

def convert_model(input_path, output_path):
    """
    Convert a trained model to the format expected by the Flask app
    
    Args:
        input_path: Path to the original model
        output_path: Path to save the converted model
    """
    try:
        # Load the model
        model = tf.keras.models.load_model(input_path)
        print(f"Model loaded successfully from {input_path}")
        
        # Save in the format expected by the Flask app
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.save(output_path, save_format='keras')
        print(f"Model converted and saved to {output_path}")
        
        # Print model summary
        model.summary()
        
        return True
    except Exception as e:
        print(f"Error converting model: {e}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert a trained model to the format expected by the Flask app')
    parser.add_argument('--input', type=str, required=True, help='Path to the original model')
    parser.add_argument('--output', type=str, default='models/fish_disease_model.keras', help='Path to save the converted model')
    
    args = parser.parse_args()
    
    # Convert the model
    convert_model(args.input, args.output)