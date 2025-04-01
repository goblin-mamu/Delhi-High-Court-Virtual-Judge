import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TrainedJudgmentPredictor:
    def __init__(self, model_dir="./fine_tuned_model/best_model"):
        """
        Initialize the trained judgment predictor
        
        Args:
            model_dir: Directory containing the trained model
        """
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.label_map = {}
        self.reverse_label_map = {}
        self.loaded = False
        
        # Try to load the model
        self.load_model()
    
    def load_model(self):
        """Load the trained model if available"""
        try:
            if os.path.exists(self.model_dir):
                print(f"Found model directory at {self.model_dir}")
                
                # Check if necessary model files exist
                config_file = os.path.join(self.model_dir, "config.json")
                model_file = os.path.join(self.model_dir, "pytorch_model.bin")
                
                if not (os.path.exists(config_file) and os.path.exists(model_file)):
                    print(f"Missing required model files in {self.model_dir}")
                    return False
                
                # Load tokenizer and model
                try:
                    print("Loading tokenizer...")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                    
                    print("Loading model...")
                    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                    
                    # Set to evaluation mode
                    self.model.eval()
                except Exception as e:
                    print(f"Error loading model or tokenizer: {str(e)}")
                    return False
                
                # Load label mapping
                label_map_path = os.path.join(os.path.dirname(self.model_dir), "label_map.txt")
                if os.path.exists(label_map_path):
                    try:
                        with open(label_map_path, "r") as f:
                            for line in f:
                                if "\t" in line:
                                    outcome, idx = line.strip().split("\t")
                                    self.label_map[outcome] = int(idx)
                                    self.reverse_label_map[int(idx)] = outcome
                    except Exception as e:
                        print(f"Error loading label mapping: {str(e)}")
                        # Create default mapping if we couldn't load it
                        default_labels = ["Allowed", "Dismissed", "Partly Allowed", "Disposed", "Settled"]
                        for idx, outcome in enumerate(default_labels):
                            self.label_map[outcome] = idx
                            self.reverse_label_map[idx] = outcome
                else:
                    print("No label mapping file found, using default mapping")
                    # Create default mapping
                    default_labels = ["Allowed", "Dismissed", "Partly Allowed", "Disposed", "Settled"]
                    for idx, outcome in enumerate(default_labels):
                        self.label_map[outcome] = idx
                        self.reverse_label_map[idx] = outcome
                
                self.loaded = True
                print(f"Successfully loaded trained model from {self.model_dir}")
                return True
        except Exception as e:
            print(f"Error loading trained model: {str(e)}")
        
        print("No trained model available. Will use heuristic prediction.")
        return False
    
    def predict(self, text):
        """
        Predict judgment outcome based on text
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.loaded:
            return None
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get predicted class
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Map to outcome
                predicted_outcome = self.reverse_label_map.get(predicted_class, "Unknown")
                
                return {
                    "prediction": predicted_outcome,
                    "confidence": confidence
                }
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

# Singleton instance
_predictor = None

def get_predictor():
    """Get or create the predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = TrainedJudgmentPredictor()
    return _predictor

def predict_with_trained_model(text):
    """Predict using the trained model if available"""
    predictor = get_predictor()
    return predictor.predict(text)