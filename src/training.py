import os
import numpy as np
import pandas as pd

class Trainner():
  def __init__(self, data_path_file="data/data-ligth.csv"):
    self.data_path_file = data_path_file

  # Load data
  def load_data(self):
    if self.data_path_file.endswith(".csv"):
      # if file exist
      if os.path.exists(self.data_path_file):
        self.data = pd.read_csv(self.data_path_file)
        print("Data loaded")
        print(self.data.head())
      else:
        raise FileNotFoundError("Data file not found")
    else:
      raise ValueError("Data file must be a CSV file")

  # Clean data
  def clean_data(self):
    pass

  # Normalize data
  def normalize_data(self):
    pass

  # Visualize data
  def visualize_data(self):
    pass

  # Split data
  def split_data(self):
    pass

  # Train models
  def train_models(self):
    pass

  # Evaluate models
  def evaluate_models(self):
    pass

  # Save models
  def save_models(self, model_path_folder="models/"):
    pass
  
  def start(self):
    self.load_data()
    self.clean_data()
    self.normalize_data()
    self.visualize_data()
    self.split_data()
    self.train_models()
    self.evaluate_models()
    self.save_models()
  
# Run training
if __name__ == '__main__':
  trainer = Trainner()
  trainer.start()