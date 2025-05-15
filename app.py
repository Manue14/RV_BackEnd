from flask import Flask
import json
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/test')
def hello():
    return {'test1': 1, 'test2': 2}

@app.get('/familias')
def get_familias():
    try:
        with open('api/familias.json', 'r', encoding='utf-8') as file:
            familias = json.load(file)
        return familias
    except FileNotFoundError:
        return {'error': 'No se ha podido leer el archivo'}, 404

@app.get('/provincias')
def get_provincias():
    try:
        with open('api/provincias.json', 'r', encoding='utf-8') as file:
            provincias = json.load(file)
        return provincias
    except FileNotFoundError:
        return {'error': 'No se ha podido leer el archivo'}, 404