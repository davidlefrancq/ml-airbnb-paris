from predictor import Predictor

# Exemple d'utilisation
if __name__ == '__main__':
    predictor = Predictor()
    
    # print("\nValeurs possibles :")
    # print(predictor.get_possible_values())
        
    # Exemples
    sample_data_list = [
      {
        'latitude': 48.833490,
        'longitude': 2.31852,
        'zipcode': 75014,
        'accommodates': 2,        
        'room_type': 'Entire home/apt',
        'beds': 0,
        'minimum_nights': 3,
      },
      {
        'latitude': 48.85100,
        'longitude': 2.35869,
        'zipcode': 75004,
        'accommodates': 2,        
        'room_type': 'Entire home/apt',
        'beds': 1,
        'minimum_nights': 1,
      },
      {
        'latitude': 48.85758,
        'longitude': 2.35275,
        'zipcode': 75004,
        'accommodates': 4,        
        'room_type': 'Entire home/apt',
        'beds': 2,
        'minimum_nights': 10,
      },

      {
        'latitude': 48.86528,
        'longitude': 2.39325,
        'zipcode': 75020,
        'accommodates': 3,        
        'room_type': 'Entire home/apt',
        'beds': 1,
        'minimum_nights': 3,
      },

      {
        'latitude': 48.85899,
        'longitude': 2.34735,
        'zipcode': 75001,
        'accommodates': 2,        
        'room_type': 'Entire home/apt',
        'beds': 1,
        'minimum_nights': 180,
      }
    ]
    
    # Predictions
    for sample_data in sample_data_list:
      print("\n")
      print(sample_data)
      predictions = predictor.predict(sample_data)
      print("Prix prédits:")
      for model_name, price in predictions.items():
        print(f"{model_name}: {price}€")