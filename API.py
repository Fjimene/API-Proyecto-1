# ============================================
# FLASK API PARA PREDECIR POPULARIDAD SPOTIFY
# Proyecto: Grupo Redes Neuronales
# ============================================

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import pandas as pd

# =========================
# Cargar modelo y objetos
# =========================
model = joblib.load('spotify_catboost_model.pkl')
columnas_modelo = joblib.load('spotify_columns.pkl')

# =========================
# Crear la API
# =========================
app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Predicción Popularidad Spotify',
    description='API para predecir la popularidad de canciones en Spotify. ' \
    'Proyecto Grupo Redes Neuronales.'
)
ns = api.namespace('predict', description='Redes Neuronales - Predicción de popularidad')

# =========================
# Definir los parámetros de entrada
# =========================
parser = ns.parser()
parser.add_argument('danceability', type=float, required=True, help='Nivel de bailabilidad (entre 0 y 1)', location='args')
parser.add_argument('energy', type=float, required=True, help='Nivel de energía de la canción (entre 0 y 1)', location='args')
parser.add_argument('loudness', type=float, required=True, help='Volumen en decibeles (valor negativo típico -60 a 0)', location='args')
parser.add_argument('speechiness', type=float, required=True, help='Contenido hablado de la canción (entre 0 y 1)', location='args')
parser.add_argument('acousticness', type=float, required=True, help='Nivel de acústica de la canción (entre 0 y 1)', location='args')
parser.add_argument('instrumentalness', type=float, required=True, help='Probabilidad de ser instrumental (entre 0 y 1)', location='args')
parser.add_argument('liveness', type=float, required=True, help='Presencia de público en vivo (entre 0 y 1)', location='args')
parser.add_argument('valence', type=float, required=True, help='Positividad de la canción (entre 0 y 1)', location='args')
parser.add_argument('tempo', type=float, required=True, help='Tempo en beats per minute (BPM)', location='args')
parser.add_argument('duration_ms', type=float, required=True, help='Duración de la canción en milisegundos', location='args')
parser.add_argument('key', type=int, required=True, help='Clave musical de la canción (0=C, 1=C♯/D♭, ..., 11=B)', location='args')
parser.add_argument('mode', type=int, required=True, help='Modalidad: 1 Mayor, 0 Menor', location='args')
parser.add_argument('time_signature', type=int, required=True, help='Compás (ejemplo: 4 para 4/4)', location='args')

# =========================
# =========================
# Modelo de salida del endpoint principal
# =========================
# Define el formato de salida que retorna el modelo
output_model = api.model('Predicción', {
    'Prediccion_popularidad': fields.Float,
})

# =========================
# Endpoint principal de predicción
# =========================
@ns.route('/')
class PopularidadAPI(Resource):
    @ns.doc(parser=parser)
    @ns.marshal_with(output_model)
    def get(self):
        # 1. Parsear los argumentos recibidos
        args = parser.parse_args()
        input_dict = {k: args[k] for k in args}

        # 2. Convertir a DataFrame y asegurar tipos numéricos
        input_df = pd.DataFrame([input_dict]).astype(float)

        # 3. Ordenar columnas como espera el modelo
        input_df = input_df[columnas_modelo]

        # 4. Hacer predicción
        pred = model.predict(input_df)[0]

        # 5. Retornar resultado
        return {'Prediccion_popularidad': pred}, 200

# =========================
# Modelo de salida del endpoint de ejemplo
# =========================
demo_output = api.model('DemoCasos', {
    'Observacion_1': fields.Raw,
    'Prediccion_1': fields.Float,
    'Popularidad_real_1': fields.Integer,
    'Observacion_2': fields.Raw,
    'Prediccion_2': fields.Float,
    'Popularidad_real_2': fields.Integer
})

# =========================
# Endpoint con 2 observaciones de ejemplo
# =========================
@ns.route('/observaciones')
class DemoObservaciones(Resource):
    @ns.marshal_with(demo_output)
    def get(self):
        # Ejemplo 1: datos simulados
        obs1 = {
            'danceability': 0.305, 'energy': 0.849, 'loudness': -10.795,
            'speechiness': 0.0549, 'acousticness': 0.000058, 'instrumentalness': 0.056700,
            'liveness': 0.4640, 'valence': 0.32, 'tempo': 141.793,
            'duration_ms': 211533, 'key': 9, 'mode': 1, 'time_signature': 4
        }

        # Ejemplo 2: datos simulados
        obs2 = {
            'danceability': 0.55, 'energy': 0.509, 'loudness': -9.661,
            'speechiness': 0.0362, 'acousticness': 0.777, 'instrumentalness': 0.202,
            'liveness': 0.1150, 'valence': 0.5440, 'tempo': 90.459,
            'duration_ms': 216506, 'key': 1, 'mode': 1, 'time_signature': 3
        }


        real_1 = 22
        real_2 = 37


        # Unir en un DataFrame
        df = pd.DataFrame([obs1, obs2]).astype(float)
        df = df[columnas_modelo]
        preds = model.predict(df)

        # Hacer predicciones
        preds = model.predict(df)

        # Devolver los inputs y sus respectivas predicciones
        return {
            'Observacion_1': obs1,
            'Prediccion_1': float(preds[0]),
            'Popularidad_real_1': 22,
            'Error_absoluto_1': round(abs(preds[0] - real_1), 2),
            'Observacion_2': obs2,
            'Prediccion_2': float(preds[1]),
            'Popularidad_real_2': 37,
            'Error_absoluto_2': round(abs(preds[1] - real_2), 2)
        }, 200
    

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)