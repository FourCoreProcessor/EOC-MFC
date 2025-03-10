import logging
# Import all necessary classes from model.py
from main import DataPrep, CNNFeatureExtractor, SVMClassifier, FacialEmotionDetector

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_log.log"),
        logging.StreamHandler()
    ]
)

def test_new_image(image_path, dataset_path="dummy_path"):
    """
    Test a new image using the pre-trained model.
    
    Args:
        image_path (str): Path to the new image to test.
        dataset_path (str): Dummy path for DataPrep initialization (not used in prediction).
    
    Returns:
        int: Predicted emotion label (e.g., 0 for happy, 1 for sad, etc.).
    """
    # Initialize the detector
    detector = FacialEmotionDetector(dataset_path)
    
    # Predict the emotion
    predicted_emotion = detector.predict_emotion(image_path)
    
    # Map the numeric prediction back to emotion name
    emotion_dict = {"happy": 0, "sad": 1, "anger": 2, "disgust": 3, "neutral": 4,
                    "surprise": 5, "contempt": 6, "fear": 7}
    emotion_name = [k for k, v in emotion_dict.items() if v == predicted_emotion][0]
    
    print(f"Predicted emotion: {emotion_name} (Label: {predicted_emotion})")
    return predicted_emotion

if __name__ == "__main__":
    # Path to your new test image
    test_image_path = "test.jpg"  # Adjust this if needed
    
    # Test the image
    test_new_image(test_image_path)