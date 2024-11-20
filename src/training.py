import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

class Trainer():
  """Machine Learning trainer for Airbnb price prediction."""

  def __init__(self, data_path_file="data/data-ligth.csv"):
    self.data_path_file = data_path_file
    self.models = {
        'random_forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1]
            }
        },
        'linear_regression': {
            'model': LinearRegression(),
            'params': {}
        }
    }
    self.scalers = {}
    self.encoders = {}
    self.best_model = None
    self.best_model_name = None
    self.best_score = float('-inf')
    self.cv_results = {}
    self.logger = self._setup_logger()

  def _setup_logger(self) -> logging.Logger:
      """Configure logging."""
      logger = logging.getLogger('MLTrainer')
      logger.setLevel(logging.INFO)
      handler = logging.StreamHandler()
      formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
      handler.setFormatter(formatter)
      logger.addHandler(handler)
      return logger

  # Load data
  def load_data(self):
    """Load CSV data file."""
    try:
      if not self.data_path_file.endswith(".csv"):
        raise ValueError("\nData file must be a CSV file")
      if not os.path.exists(self.data_path_file):
        raise FileNotFoundError("\nData file not found")
      self.data = pd.read_csv(self.data_path_file)
      self.logger.info(f"Data from {self.data_path_file} loaded successfully: {self.data.shape}")
    except Exception as e:
      self.logger.error(f"Error: loading data has failed from file: {self.data_path_file}")
      raise e

  # Clean data
  def clean_data(self):
    try:
      self.data.info()
      # colonne conservé: neighbourhood, latitude, longitude, room_type, price, minimum_nights
      colomns_used = ['neighbourhood', 'latitude', 'longitude', 'room_type', 'price', 'minimum_nights']
      
      # retirer les colonnes non utilisées
      self.data = self.data[colomns_used]
        
      # retirer les lignes avec le price manquant
      self.data = self.data.dropna(subset=['price'])
      
      # supprimer tous les caractère non numérique à l'exception du "."
      if not np.issubdtype(self.data['price'].dtype, np.number):
        self.data['price'] = self.data['price'].str.replace(r'[^\d.]', '', regex=True).astype(float)

      # Identifier et supprimer les valeurs aberrantes
      for column in ['price', 'minimum_nights']:
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        self.data = self.data[
          (self.data[column] >= Q1 - 1.5 * IQR) & 
          (self.data[column] <= Q3 + 1.5 * IQR)
        ]
      self.logger.info(f"Data cleaned successfully: {self.data.shape}")
      self.data.info()
    except Exception as e:
      self.logger.error(f"Error: cleaning data has failed.")
      raise e

  # Normalize data
  def normalize_data(self):
    """Normalize numerical features and encode categorical features."""

    try:
      # Encoder les variables catégorielles
      categorical_columns = ['neighbourhood', 'room_type']
      for column in categorical_columns:
        self.encoders[column] = LabelEncoder()
        self.data[column] = self.encoders[column].fit_transform(self.data[column])
          
      self.data.info()
      
      # Standardiser les variables numériques
      numerical_columns = ['price', 'latitude', 'longitude', 'minimum_nights']
      for column in numerical_columns:
        self.scalers[column] = StandardScaler()
        self.data[column] = self.scalers[column].fit_transform(self.data[[column]])
      
      self.logger.info("Data normalization completed")

      # Save encoders and scalers
      for name, encoder in self.encoders.items():
        joblib.dump(encoder, os.path.join('models', f'encoder_{name}.pkl'))
      for name, scaler in self.scalers.items():
        joblib.dump(scaler, os.path.join('models', f'scaler_{name}.pkl'))

      # Vérifier les statistiques descriptives des variables normalisées
      print("\nStatistiques descriptives après normalisation:")
      print(self.data.describe())
    except Exception as e:
        self.logger.error(f"Error: normalizing data has failed.")
        raise e

  # Visualize data
  def visualize_data(self):
    """Generate and save data visualizations."""
    try:
      plots = {
        'price_distribution': lambda: sns.histplot(data=self.data, x="price", kde=True),
        'minimum_nights_distribution': lambda: sns.histplot(data=self.data, x="minimum_nights", kde=True),
        'price_by_room_type': lambda: sns.boxplot(data=self.data, x="room_type", y="price"),
        'price_by_location': lambda: sns.scatterplot(data=self.data, x="longitude", y="latitude", hue="price"),
        'correlation_matrix': lambda: sns.heatmap(self.data.corr(), annot=True, cmap="YlOrRd")
      }
      
      for name, plot_func in plots.items():
        plt.figure(figsize=(10, 6))
        plot_func()
        plt.title(name.replace('_', ' ').title())
        plt.tight_layout()
        plt.savefig(os.path.join('plots', f'{name}.png'))
        plt.close()
          
      self.logger.info("Data visualizations saved")
        
    except Exception as e:
      self.logger.error(f"Error: generating visualizations has failed.")
      raise e

  # Split data
  def split_data(self):
    """Split data into training and testing sets."""
    try:
      features = ['neighbourhood', 'latitude', 'longitude', 'room_type', 'minimum_nights']
      X = self.data[features]
      y = self.data['price']
      
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
      )
      
      self.logger.info(f"Data split - Training: {self.X_train.shape}, Testing: {self.X_test.shape}")
      
    except Exception as e:
      self.logger.error(f"Error: splitting data has failed.")
      raise e

  # Train models
  def train_models(self):
    """Train models with hyperparameter optimization."""
    try:
      for name, config in self.models.items():
        if config['params']:
          grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
          )
          grid_search.fit(self.X_train, self.y_train)
          self.models[name]['best_model'] = grid_search.best_estimator_
          self.logger.info(f"\nBest parameters for {name}:")
          self.logger.info(grid_search.best_params_)
          self.logger.info(f"Best score: {-grid_search.best_score_:.3f}")
        else:
          config['model'].fit(self.X_train, self.y_train)
          self.models[name]['best_model'] = config['model']
              
      self.logger.info("Models trained successfully")
        
    except Exception as e:
      self.logger.error(f"Error training models: {str(e)}")
      raise

  # Evaluate models
  def evaluate_models(self):
    """Evaluate model performance."""
    try:
      for name, config in self.models.items():
        model = config['best_model']
        y_pred = model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        self.logger.info(f"\n{name} Results:")
        self.logger.info(f"RMSE: {rmse:.2f}")
        self.logger.info(f"R2 Score: {r2:.2f}")
        
        if r2 > self.best_score:
          self.best_score = r2
          self.best_model = model
          self.best_model_name = name
                
    except Exception as e:
      self.logger.error(f"Error: evaluating models has failed.")
      raise e

  # Save models
  def save_models(self, model_path_folder="models/"):
    """Save trained models."""
    try:
      for name, config in self.models.items():
        model_path = os.path.join('models', f'{name}.pkl')
        joblib.dump(config['best_model'], model_path)
        
      self.logger.info(f"Models saved in /models")
      
    except Exception as e:
      self.logger.error(f"Error: saving models has failed.")
      raise e
  
  def start(self):
      """Run the complete training pipeline."""
      pipeline_steps = [
        self.load_data,
        self.clean_data,
        self.normalize_data,
        self.visualize_data,
        self.split_data,
        self.train_models,
        self.evaluate_models,
        self.save_models
      ]
      
      for step in pipeline_steps:
        try:
          step()
        except Exception as e:
          self.logger.error(f"Error: pipeline failed at {step.__name__}.")
          raise e
  
# Run training
if __name__ == '__main__':
  trainer = Trainer()
  trainer.start()