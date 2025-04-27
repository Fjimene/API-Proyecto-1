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
# Definición de salida
# =========================
resource_fields = api.model('Resource', {
    'Prediccion_popularidad': fields.Float,
})

# =========================
# Definir el recurso
# =========================
@ns.route('/')
class PopularidadAPI(Resource):
    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()

        # Preparar input
        input_dict = {k: args[k] for k in args}
        input_data = pd.DataFrame([input_dict])

        # Asegurarse que las columnas numéricas sean floats
        input_data = input_data.astype(float)

        # Reordenar las columnas al orden original del modelo
        input_data = input_data[columnas_modelo]

        # Predicción
        prediccion = model.predict(input_data)[0]

        return {'Prediccion_popularidad': prediccion}, 200
    

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)