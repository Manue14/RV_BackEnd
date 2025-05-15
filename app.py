from flask import Flask
import json
from flask_cors import CORS
import os
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
        file_path = os.path.join(app.root_path, 'api', 'provincias.json')
        with open('api/provincias.json', 'r', encoding='utf-8-sig') as file:
            provincias = json.load(file)
        return provincias
    except (FileNotFoundError, json.JSONDecodeError) as e:
        app.logger.error(f"Error leyendo {file_path}: {e}")
        return {'error': str(e)}, 500
    except FileNotFoundError:
        return {'error': 'No se ha podido leer el archivo'}, 404