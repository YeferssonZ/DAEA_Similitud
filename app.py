from flask import Flask, request, jsonify
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Función para obtener datos de la API de películas
def obtener_peliculas():
    url_peliculas = "http://192.168.1.103:5283/api/pelicula"
    response = requests.get(url_peliculas)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None

# Función para obtener datos de la API de géneros
def obtener_generos():
    url_generos = "http://192.168.1.103:5283/api/genero"
    response = requests.get(url_generos)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None

# Función para obtener datos de la API de calificaciones
def obtener_calificaciones():
    url_calificaciones = "http://192.168.1.103:5283/api/rating"
    response = requests.get(url_calificaciones)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None

# Codificación de los géneros de las películas usando one-hot encoding
def codificar_generos(peliculas, generos):
    genero_dict = {genero['nombre']: i for i, genero in enumerate(generos)}
    for pelicula in peliculas:
        generos_pelicula = pelicula['generos']
        generos_encoded = [0] * len(generos)
        for genero in generos_pelicula:
            genero_nombre = genero['nombre']
            if genero_nombre in genero_dict:
                genero_index = genero_dict[genero_nombre]
                generos_encoded[genero_index] = 1
        pelicula['generos_encoded'] = generos_encoded

# Función para calcular la similitud ponderada entre películas
def calcular_similitud_ponderada(pelicula_1, pelicula_2, calificacion):
    # Calcular la similitud coseno
    similitud_coseno = cosine_similarity([pelicula_1], [pelicula_2])[0][0]
    # Ponderar la similitud por la calificación
    similitud_ponderada = similitud_coseno * calificacion
    return similitud_ponderada

# Función para recomendar películas similares a una película dada
def recomendar_peliculas_similares(peliculas, calificaciones, usuario_id):
    # Obtener las calificaciones del usuario
    calificaciones_usuario = [cal for cal in calificaciones if cal['usuarioId'] == usuario_id]

    if len(calificaciones_usuario) < 2:
        return None

    # Ordenar las calificaciones por timestamp (más recientes primero)
    calificaciones_usuario.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

    # Obtener las películas correspondientes a estas calificaciones
    peliculas_referencia = []
    for calificacion in calificaciones_usuario:
        pelicula = next((p for p in peliculas if p['id'] == calificacion['peliculaId']), None)
        if pelicula:
            peliculas_referencia.append((pelicula, calificacion['calificacion']))

    if len(peliculas_referencia) < 2:
        return None

    # Calcular la similitud de todas las demás películas con las películas de referencia
    similitudes = []
    for otra_pelicula in peliculas:
        if otra_pelicula['id'] in [p['id'] for p, _ in peliculas_referencia]:
            continue  # Excluir películas ya vistas

        # Calcular la similitud ponderada con todas las películas de referencia
        similitud_total = 0
        for pelicula_referencia, calificacion in peliculas_referencia:
            similitud_total += calcular_similitud_ponderada(pelicula_referencia['generos_encoded'], otra_pelicula['generos_encoded'], calificacion)

        # Normalizar la similitud total
        similitud_total /= len(peliculas_referencia)
        similitudes.append((otra_pelicula, similitud_total))

    similitudes.sort(key=lambda x: x[1], reverse=True)

    if similitudes:
        return similitudes[0][0]
    else:
        return None

@app.route('/recomendar/<usuario_id>', methods=['GET'])
def recomendar_peliculas(usuario_id):
    print(f"Recibida solicitud de recomendación para usuario: {usuario_id}")
    # Obtener datos de las APIs
    peliculas = obtener_peliculas()
    generos = obtener_generos()
    calificaciones = obtener_calificaciones()

    if peliculas and generos and calificaciones:
        # Codificar los géneros de las películas utilizando one-hot encoding
        codificar_generos(peliculas, generos)

        # Imprimir las calificaciones del usuario
        calificaciones_usuario = [cal for cal in calificaciones if cal['usuarioId'] == usuario_id]
        calificaciones_usuario.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        print(f"Calificaciones del usuario: {calificaciones_usuario}")

        # Recomendar una película similar basada en las calificaciones del usuario
        pelicula_recomendada = recomendar_peliculas_similares(peliculas, calificaciones, usuario_id)

        if pelicula_recomendada:
            print(f"Película recomendada: {pelicula_recomendada}")
            pelicula_data = {
                'id': pelicula_recomendada['id'],
                'titulo': pelicula_recomendada['titulo'],
                'videoUrl': pelicula_recomendada['videoUrl'],
                'generos': pelicula_recomendada['generos']
            }
            return jsonify({'pelicula_recomendada': pelicula_data})
        else:
            return jsonify({'error': 'No se pudo encontrar una película recomendada'})
    else:
        return jsonify({'error': 'No se pudieron obtener datos de las APIs'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4000)
