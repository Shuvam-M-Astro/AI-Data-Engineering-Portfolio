#!/usr/bin/env python3
"""
Setup script for AI-Data-Engineering-Portfolio.
Installs common dependencies and project-specific requirements.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def install_requirements():
    """Install all requirements."""
    # Install common requirements first
    if not run_command("pip install -r requirements-common.txt", "Installing common dependencies"):
        return False

    # Get all project directories
    project_dirs = [
        "CIFAR-10",
        "RPA Sentiment Analysis",
        "LangChain",
        "RAG",
        "MLOps",
        "ModelServing",
        "SpeechRecognition",
        "MNIST",
        "IMDB_Consolidated",
    ]

    # Install project-specific requirements
    for project_dir in project_dirs:
        req_file = Path(project_dir) / "requirements.txt"
        if req_file.exists():
            if not run_command(
                f"pip install -r {req_file}", f"Installing {project_dir} dependencies"
            ):
                return False

    return True


def create_environment():
    """Create virtual environment if needed."""
    if not os.path.exists("venv"):
        print("🔄 Creating virtual environment...")
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
        print("✅ Virtual environment created")
        print("📝 To activate the environment:")
        if sys.platform == "win32":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
    else:
        print("✅ Virtual environment already exists")


def main():
    """Main setup function."""
    print("🚀 Setting up AI-Data-Engineering-Portfolio...")

    # Check if we're in the right directory
    if not os.path.exists("requirements-common.txt"):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)

    # Create virtual environment
    create_environment()

    # Install requirements
    if install_requirements():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Activate the virtual environment")
        print("2. Navigate to any project directory")
        print("3. Run the project-specific scripts")
        print("\n📚 Available projects:")
        print("- CIFAR-10: Image classification")
        print("- MNIST: Digit recognition with A/B testing")
        print("- IRIS: Flower classification")
        print("- Titanic: Survival prediction")
        print("- IMDB_Consolidated: Sentiment analysis")
        print("- SpeechRecognition: Audio processing")
        print("- LangChain: Chatbot with LangChain")
        print("- RAG: Retrieval-Augmented Generation")
        print("- MLOps: ML pipeline management")
        print("- ModelServing: FastAPI model serving")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
