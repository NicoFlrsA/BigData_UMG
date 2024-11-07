import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.preprocessing import LabelEncoder

class KNN:
    def __init__(self, data_path, k=3, metric="euclidean"):
        # Cargar datos desde un archivo CSV
        self.data = pd.read_csv(data_path)
        
        # Codificar todas las columnas no numéricas en el dataset
        self.label_encoders = {}
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column].fillna('Unknown'))
                self.label_encoders[column] = le  # Guardar el codificador para referencia

        self.k = k
        self.metric = metric

    def fit(self):
        # Este método puede ser implementado si es necesario para entrenar el modelo
        pass

    def predict(self, anime_name):
        # Verificar si el anime_name está en los datos
        if anime_name not in self.label_encoders['title'].inverse_transform(self.data['title']):
            raise ValueError(f"{anime_name} no se encuentra en los datos.")

        # Obtener las características del anime seleccionado
        selected_anime = self.data[self.data['title'] == self.label_encoders['title'].transform([anime_name])[0]].iloc[0]
        selected_features = selected_anime[['episodes', 'type']].values.reshape(1, -1)

        # Calcular las distancias con todos los demás animes
        features = self.data[['episodes', 'type']].values
        if self.metric == "euclidean":
            distances = euclidean_distances(selected_features, features)
        elif self.metric == "cosine":
            distances = cosine_distances(selected_features, features)
        else:
            raise ValueError("Métrica no soportada")

        # Obtener los índices de los k animes más cercanos
        closest_indices = distances.argsort()[0][:self.k]

        # Obtener los títulos de los k animes recomendados
        recommended_animes = self.label_encoders['title'].inverse_transform(self.data.iloc[closest_indices]['title'])
        
        return recommended_animes.tolist()
