from fastapi import FastAPI, Request, Form
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
import logging

# Configuración del log para escribir en el archivo de logs
logging.basicConfig(
    filename="api_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Inicializar FastAPI y el modelo de IA
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Usamos un modelo pre-entrenado de Hugging Face
model_name = "finiteautomata/beto-sentiment-analysis"
sentiment_pipe = pipeline("sentiment-analysis", model=model_name)


# Modelo de datos para la API REST
class AnalysisRequest(BaseModel):
    text: str
# Logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("La API de Sentimientos está iniciando...")
    yield
    logging.info("La API de Sentimientos se está cerrando...")


# ENDPOINT 1: API REST (Recibe JSON)
@app.post("/analyze")
async def analyze_text(data: AnalysisRequest):

    try:
        # Tu lógica de predicción aquí
        result = sentiment_pipe(data.text)[0]

        logging.info(f"Procesando texto: {data.text[:50]}...") 
        logging.info(f"Resultado: {result['label']} con confianza {result['score']:.4f}")
        return {"text": data.text, "label": result['label'], 
                "score": result['score'],
                "status": "ok"}
    except Exception as e:
        logging.error(f"Error en la predicción: {str(e)}")
        return {"error": "Error interno al procesar el modelo"}
# ENDPOINT 2: Cliente Jinja (Interfaz Web)
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/web-analyze")
def web_analyze(request: Request, text: str = Form(...)):
    # Consumimos la lógica del modelo directamente para la web
    try:
        result = sentiment_pipe(text)[0]
        logging.info(f"Procesando texto: {text[:50]}...") 
        logging.info(f"Resultado: {result['label']} con confianza {result['score']:.4f}")
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "text": text, 
            "result": result['label'],
            "confidence": f"{result['score']:.4%}"
        })
    except Exception as e:
        logging.error(f"Error en la predicción web: {str(e)}")  
        return {"error": "Error interno al procesar el modelo"}

# ENDPOINT 3: Cliente Jinja (Interfaz Web para preguntas)
@app.get("/encuesta")
def survey(request: Request):
    return templates.TemplateResponse("index1.html", {"request": request})  

@app.post("/web-analyze-preguntas")
def web_analyze_questions(
    request: Request, 
    p1: str = Form(...), 
    p2: str = Form(...), 
    p3: str = Form(...)
):
    preguntas = [p1, p2, p3]
    resultados = []
    try:
        for pregunta in preguntas:
            result = sentiment_pipe(pregunta)[0]
            resultados.append({
                "label": result['label'],
                "score": result['score'],
                "question": pregunta
            })
            logging.info(f"Procesando pregunta: {pregunta[:50]}...") 
            logging.info(f"Resultado: {result['label']} con confianza {result['score']:.4f}")
        
        return templates.TemplateResponse("index1.html", {
            "request": request, 
            "resultados": resultados
        })
    except Exception as e:
        logging.error(f"Error en la predicción de preguntas: {str(e)}")  
        return templates.TemplateResponse("index1.html", {  
            "request": request, 
            "error_msg": "Error interno al procesar el modelo"
        })  


# ENDPOINT 4: Concatenar pregunta y respuesta para brindar contexto
@app.get("/contexto")
def context(request: Request):
    return templates.TemplateResponse("index3.html", {"request": request})

@app.post("/web-analyze-context")
def web_analyze_context(        
    request: Request, 
    p1: str = Form(...), 
    p2: str = Form(...), 
    p3: str = Form(...),
    r1: str = Form(...), 
    r2: str = Form(...), 
    r3: str = Form(...)
):
    # Lista de pares concatenados para procesar
    entradas = [
        f"{p1} [SEP] {r1}", # Usando el token SEP de BERT para separar contexto
        f"{p2} [SEP] {r2}",
        f"{p3} [SEP] {r3}"
    ]
    
    resultados = []
    try:
        for entrada in entradas:
            result = sentiment_pipe(entrada)[0]
            pregunta = entrada.split(" [SEP] ")[0]
            respuesta = entrada.split(" [SEP] ")[1]
            resultados.append({
                "label": result['label'],
                "score": result['score'],
                "pregunta": pregunta,
                "respuesta": respuesta
            })
            logging.info(f"Procesando contexto: {entrada[:100]}...") 
            logging.info(f"Resultado: {result['label']} con confianza {result['score']:.4f}")
        return templates.TemplateResponse("index3.html", {
            "request": request, 
            "resultados": resultados
        })
    except Exception as e:
        logging.error(f"Error en la predicción de contexto: {str(e)}")  
        return templates.TemplateResponse("index3.html", {  
            "request": request,
            "error_msg": "Error interno al procesar el modelo"
        })