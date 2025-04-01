import subprocess
import sys
import os
import time

def install_packages():
    """Install required packages"""
    print("Installing required packages...")
    packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "torch",
        "transformers",
        "tqdm",
        "beautifulsoup4",
        "requests"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            # Add the --no-cache-dir flag to avoid caching issues
            # Add the --no-deps flag to avoid dependency conflicts
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", package])
            print(f"Successfully installed {package}")
        except Exception as e:
            print(f"Warning: Could not install {package}: {str(e)}")
            print(f"Continuing without {package}. Some functionality may be limited.")
    
    print("Package installation completed")

def run_data_collection():
    """Run data collection script"""
    print("\n=== STEP 1: COLLECTING DATA ===")
    subprocess.check_call([sys.executable, "data_downloader.py"])

def run_model_training():
    """Run model training script"""
    print("\n=== STEP 2: TRAINING MODEL ===")
    
    # Check if data file exists
    data_path = "./data/processed/dhc_judgments_sample.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return False
    
    # Run training with default parameters
    subprocess.check_call([
        sys.executable, 
        "train_model.py",
        "--batch-size", "2",  # Small batch size for CPU
        "--epochs", "2",      # Fewer epochs for faster training
        "--max-length", "128" # Shorter sequences for less memory usage
    ])
    
    return True

def integrate_model():
    """Integrate trained model with application"""
    print("\n=== STEP 3: INTEGRATING MODEL WITH APPLICATION ===")
    
    model_path = "./fine_tuned_model/best_model"
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        return False
    
    # Update the application config to use the trained model
    config = {
        "model_path": model_path,
        "trained_date": time.strftime("%Y-%m-%d"),
        "model_type": "distilbert"  # The model we used
    }
    
    # Create model config file
    os.makedirs("./config", exist_ok=True)
    with open("./config/model_config.txt", "w") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")
    
    print(f"Model integrated successfully. Configuration saved to ./config/model_config.txt")
    return True

def main():
    try:
        # Step 0: Install required packages
        install_packages()
        
        # Step 1: Collect data
        run_data_collection()
        
        # Step 2: Train model
        if run_model_training():
            # Step 3: Integrate model
            integrate_model()
            
            print("\n=== TRAINING PROCESS COMPLETE ===")
            print("You can now restart the application to use the trained model.")
        
    except Exception as e:
        print(f"Error during training process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()