"""
Download required NLTK data for text analysis
Run this once after installing dependencies
"""
import nltk

print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
print("NLTK data downloaded successfully!")
