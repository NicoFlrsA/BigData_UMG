# Anime Recommendation API

This project is a Flask-based API that provides anime recommendations using a K-Nearest Neighbors (KNN) model. You can input the name of an anime and get a list of recommended titles based on your input.

![image](https://github.com/user-attachments/assets/2b5a34bd-f879-4cd5-b262-17f8eacce6c3)

## Features
- **Route `/`:** Displays a welcome message for the API.
- **Route `/recommended/<anime>`:** Accepts an anime name as input and returns a JSON response with recommendations.

## Prerequisites
Before running the API, make sure you have the following installed:

- Python 3.7+
- Flask library
- Required dependencies listed in the `requirements.txt`
- The `AnimeList.csv` dataset file located in the `../data/raw/` directory relative to the project folder.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify that the `AnimeList.csv` file is in the correct directory:
   ```
   ../data/raw/AnimeList.csv
   ```

## Running the API
1. Save the following code as `app.py`:

```python
from flask import Flask, jsonify
from NFA_KNN import KNN

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Welcome to the Anime Recommendation API!</p>"

@app.route("/recommended/<anime>")
def recommended(anime):
    # Initialize the KNN model
    model = KNN("../data/raw/AnimeList.csv", k=3, metric="euclidean")

    try:
        # Get the recommendations
        recommended_animes = model.predict(anime)
        
        # Format recommendations as JSON
        data = {
            "anime": anime,
            "recommended_animes": recommended_animes
        }
        return jsonify(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
```

2. Run the API:
   ```bash
   python app.py
   ```

3. Open your browser or API client (e.g., Postman) and navigate to:
   - [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to see the welcome message.
   - [http://127.0.0.1:5000/recommended/<anime>](http://127.0.0.1:5000/recommended/<anime>) to get anime recommendations (replace `<anime>` with an anime name).

## Example
**Request:**
```http
GET http://127.0.0.1:5000/recommended/Naruto
```

**Response:**
```json
{
  "anime": "Naruto",
  "recommended_animes": [
    "Naruto Shippuden",
    "Bleach",
    "One Piece"
  ]
}
```

## Notes
- Ensure that the `NFA_KNN` module and its dependencies are correctly implemented.
- The `AnimeList.csv` dataset should be clean and formatted appropriately for the KNN model.
- Debugging mode (`debug=True`) is only recommended for development purposes. Disable it in production.

## License
This project is licensed under the [MIT License](LICENSE).

---

Feel free to customize this README to better fit your needs!

