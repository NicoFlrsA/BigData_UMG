from flask import Flask, jsonify
from NFA_KNN import KNN

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Welcome to the Anime Recommendation API!</p>"

@app.route("/recommended/<anime>")
def recommended(anime):
    # Inicializa el modelo KNN
    model = KNN("../data/raw/AnimeList.csv", k=3, metric="euclidean")

    try:
        # Obtener las recomendaciones
        recommended_animes = model.predict(anime)
        
        # Formatear las recomendaciones en un formato JSON
        data = {
            "anime": anime,
            "recommended_animes": recommended_animes
        }
        return jsonify(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
