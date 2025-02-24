from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from transformers import pipeline



app = Flask(__name__)
app.secret_key = 'secret'

qa_pipeline = pipeline("question-answering")

UPLOAD_FOLDER = 'entorno'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def limpiar_comentarios(texto):
    """Elimina emojis y caracteres no deseados del texto"""
    texto_limpio = re.sub(r'[^\w\s,¿?!¡]', '', texto)  
    texto_limpio = re.sub(r'[^\x00-\x7F]+', '', texto_limpio)  
    return texto_limpio

def entrenar_modelo():
    """Entrena el modelo de clasificación utilizando un conjunto de datos etiquetados."""
    data = [
        {"comentario": "Noboa es un gran líder, el mejor para el país", "etiqueta": "Voto Noboa"},
        {"comentario": "¡Viva el presidente Noboa, el país necesita su liderazgo!", "etiqueta": "Voto Noboa"},
        {"comentario": "Borrego, Noboa es la mejor opción para Ecuador", "etiqueta": "Voto Noboa"},
        {"comentario": "No quiero votar por Luisa, es una mentira de Correa", "etiqueta": "Voto Noboa"},
        {"comentario": "Luisa es una excelente candidata, pero Noboa es mi elección", "etiqueta": "Voto Noboa"},
        {"comentario": "Florindo es un excelente candidato, siempre lo he admirado", "etiqueta": "Voto Luisa"},
        {"comentario": "Correa ha hecho mucho por Ecuador, Luisa es su sucesora", "etiqueta": "Voto Luisa"},
        {"comentario": "Luisa tiene grandes propuestas para el país", "etiqueta": "Voto Luisa"},
        {"comentario": "Cartón, Luisa es la única opción con visión de futuro", "etiqueta": "Voto Luisa"},
        {"comentario": "El presidente Correa siempre estuvo del lado del pueblo", "etiqueta": "Voto Luisa"},
        {"comentario": "¡Luisa presidenta! Ella sí sabe cómo sacar adelante a Ecuador", "etiqueta": "Voto Luisa"},
        {"comentario": "Ninguno de los dos me convence, el futuro de Ecuador está en duda", "etiqueta": "Voto Nulo"},
        {"comentario": "No confío ni en Noboa ni en Luisa, el país merece algo mejor", "etiqueta": "Voto Nulo"},
        {"comentario": "Me da igual, al final todos son lo mismo, no votaré", "etiqueta": "Voto Nulo"},
        {"comentario": "No quiero votar por ellos, no me representan", "etiqueta": "Voto Nulo"},
        {"comentario": "No tengo clara ninguna de las opciones", "etiqueta": "Voto Nulo"},
    ]

    df = pd.DataFrame(data)
    X = df['comentario']  
    y = df['etiqueta']  
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    
   
    accuracy = model.score(X_test, y_test)
    print(f"Precisión del modelo: {accuracy * 100:.2f}%")
    
    return model


modelo_clasificacion = entrenar_modelo()

def clasificar_comentario(texto):
    """Clasifica el comentario utilizando el modelo entrenado."""
    texto = limpiar_comentarios(texto)
    resultado = modelo_clasificacion.predict([texto])[0]
    return resultado

def analizar_comentarios(filepath):
    try:
        df = pd.read_excel(filepath)

        
        if 'text' not in df.columns:
            return "Error: El archivo no contiene la columna 'text'", 400

        
        df['text'] = df['text'].apply(limpiar_comentarios)

        
        df['Clasificacion'] = df['text'].apply(clasificar_comentario)

       
        resultado_path = os.path.join(UPLOAD_FOLDER, 'resultado.xlsx')
        df.to_excel(resultado_path, index=False)
        return df
    except Exception as e:
        return f"Error al procesar el archivo: {str(e)}", 500

def responder_pregunta(pregunta, contexto):
    """Responde preguntas sobre los datos utilizando un modelo NLP"""
    result = qa_pipeline({'question': pregunta, 'context': contexto})
    return result['answer']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/procesar', methods=['POST'])
def procesar():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if not file.filename.endswith('.xlsx'):
        return "Error: El archivo debe ser un archivo Excel (.xlsx)", 400
    
    # Validar tipo de archivo
    if file.mimetype != 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        return "Error: El archivo debe ser un archivo Excel válido.", 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Analizar el archivo
    df_resultado = analizar_comentarios(filepath)
    
    if isinstance(df_resultado, tuple):  
        return df_resultado[0], df_resultado[1]
    
    # Generar gráfico
    generar_grafico(df_resultado)

    # Guardar en la sesión
    session['file'] = file.filename
    return redirect(url_for('resultados'))

def generar_grafico(df):
    try:
        conteo = df['Clasificacion'].value_counts()
        plt.figure(figsize=(8,6))
        conteo.plot(kind='bar', color=['blue', 'green', 'red'])
        plt.title('Resultados Electorales')
        plt.xlabel('Opción')
        plt.ylabel('Cantidad de votos')
        plt.xticks(rotation=45)

        # Guardar el gráfico en la carpeta 'static'
        static_path = os.path.join('static', 'grafico.png')
        plt.savefig(static_path)
    except Exception as e:
        print(f"Error al generar el gráfico: {str(e)}")

@app.route('/resultados')
def resultados():
    # Abre el archivo Excel y lee los datos con pandas
    df = pd.read_excel('entorno/resultado.xlsx')

    # Pasa los datos a la plantilla
    comentarios = df.to_dict(orient='records')  # Convierte el DataFrame en una lista de diccionarios
    grafico = 'static/grafico.png'  # Ruta del gráfico
    
    return render_template('resultados.html', comentarios=comentarios, grafico=grafico)

@app.route('/chat', methods=['POST'])
def chat():
    """Interacción del chat con el usuario"""
    pregunta = request.form.get('pregunta')
    if not pregunta:
        return redirect(url_for('index'))

    # Leemos el archivo Excel y obtenemos los comentarios
    df = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], session.get('file')))
    contexto = df.to_string()  # Convertimos el DataFrame a texto para que el modelo lo use como contexto

    # Procesamos la pregunta
    respuesta = responder_pregunta(pregunta, contexto)
    
    return render_template('chat.html', pregunta=pregunta, respuesta=respuesta)

@app.route('/obtener_respuesta', methods=['POST'])
def obtener_respuesta():
    data = request.get_json()
    pregunta = data.get('pregunta')
    contexto = "Aquí iría el contexto de los comentarios relevantes para las respuestas."
    
    respuesta = qa_pipeline({
        'context': contexto,
        'question': pregunta
    })
    
    return jsonify({'respuesta': respuesta['answer']})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
