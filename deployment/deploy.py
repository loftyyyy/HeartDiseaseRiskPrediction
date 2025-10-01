"""
Deployment Script for Heart Disease Risk Prediction
This script handles the complete deployment process
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def train_model():
    """Train the model."""
    print("Training the model...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Model trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error training model: {e}")
        return False

def run_app():
    """Run the Streamlit application."""
    print("Starting Streamlit application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main deployment function."""
    print("=" * 60)
    print("🚀 HEART DISEASE RISK PREDICTION - DEPLOYMENT")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ Please run this script from the deployment directory")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Train model
    if not train_model():
        return
    
    print("\n" + "=" * 60)
    print("🎉 DEPLOYMENT READY!")
    print("=" * 60)
    print("The application will start in your browser...")
    print("Press Ctrl+C to stop the application")
    print("=" * 60)
    
    # Run the application
    run_app()

if __name__ == "__main__":
    main()

