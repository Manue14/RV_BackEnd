from flask import Flask, request, make_response, jsonify
import json
from flask_cors import CORS, cross_origin
import pandas as pd
import os
app = Flask(__name__)
CORS(
    app,
    resources={ r"/*": { "origins": "*" } },
    supports_credentials=False
)

@app.get('/tiendas')
def get_tiendas():
    try:
        file_path = os.path.join(app.root_path, 'api', 'top_tiendas.json')
        with open('api/top_tiendas.json', 'r', encoding='utf-8-sig') as file:
            top_tiendas = json.load(file)
        return list(top_tiendas.keys())

    except (FileNotFoundError, json.JSONDecodeError) as e:
        app.logger.error(f"Error leyendo {file_path}: {e}")
        return {'error': str(e)}, 500
    except FileNotFoundError:
        return {'error': 'No se ha podido leer el archivo'}, 404

@app.get('/datos_tienda')
def post_tiendas():
    try:
        tienda_id = request.args.get("idTienda")
        print(tienda_id)
        with open('api/top_tiendas.json', 'r', encoding='utf-8') as file:
            tiendas = json.load(file)
        tienda = tiendas[tienda_id]
        with open('api/familias.json', 'r', encoding='utf-8') as file:
            familias = json.load(file)
        datos_tienda = {}
        datos_tienda["provincia"] = tienda["provincia"]
        datos_tienda["codigo_postal"] = tienda["codigo_postal"]
        datos_tienda["productos"] = dict()
        for key in familias:
            if int(key) in tienda["productos"]:
                datos_tienda["productos"][key] = familias[key]
        return datos_tienda

    except FileNotFoundError:
        return {'error': 'No se ha podido leer el archivo'}, 404

@app.post("/predict")
def predict():
    try:
        data = request.get_json(force=True)
        print("Datos recibidos:", data)
        return jsonify({
            'status': 'success',
            'message': 'Datos recibidos correctamente',
            'data': data
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400