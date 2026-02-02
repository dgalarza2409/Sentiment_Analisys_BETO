# Teoría y desarrollo práctico de un proyecto de Análisis de sentimientos utilizando FastApi, Uvicorn, Jinja2 templates 

### Un modelo de IA para utilizar de una manera eficiente los LLM (Large Language Model) previamente entrenados con inmensos datasets, y de esta manera evitar el alto costo de un desarrollo desde cero.

Se escogio se modelo BETO por las siguientes razones (NOTA. El texto a continuación fue generado por la IA) :

  - Es la versión en **español** del modelo BERT, desarrollada por el Centro de Investigación de la Web de la Universidad de Chile.
    - Entrenamiento nativo en Español
    - Costo: Totalmente gratuito para descargar y usar.
    - Licencia: Utiliza generalmente una licencia CC BY 4.0, lo que permite su uso y adaptación citando a los autores.
    - Dónde encontrarlo: Está disponible en plataformas como Hugging Face para que desarrolladores lo integren en sus proyectos.  
  - Captura de Matices y Contexto
    - BETO utiliza una técnica llamada Whole Word Masking (WWM), que le obliga a predecir palabras completas en lugar de solo fragmentos.
    - En el análisis de sentimientos, esto es vital para identificar: 
      - Sarcasmo e ironía: Entiende el contexto de la frase completa para detectar si una palabra positiva se usa con intención negativa.
      - Negaciones: Procesa la relación entre palabras (como "no estoy feliz") de forma bidireccional, evitando clasificaciones erróneas comunes en modelos más simples.
  - Rendimiento Superior (Benchmarks)
    - En diversas competencias de procesamiento de lenguaje natural (NLP) aplicadas al español, BETO ha demostrado superar al BERT original:
      - Precisión: Se han registrado casos donde BETO alcanza un 90.95% de precisión frente al 45.73% de modelos multilingües en tareas de detección de desinformación y sentimientos.
      - Líder en tareas específicas: Sigue siendo el estándar a batir en tareas de etiquetado de partes de la oración (POS) y reconocimiento de entidades (NER), fundamentales para desglosar "quién siente qué"
  - Transfer Learning Eficiente
    - El proyecto no incluye el Transfer Learning que consiste en realizar un ajuste fino (fine-tuning) con un conjunto de datos pequeño de los sentimientos específicos (ej. reseñas de productos, comentarios de Twitter) para obtener resultados profesionales sin necesidad de un entrenamiento costoso.
    -  El modelo específico seleccionado es "finiteautomata/beto-sentiment-analysis" el mismo que es el resultado de un proceso de Transfer Learning / Fine-Tuning realizado por grupos especializados que le añadieron la capa de clasificación y lo entrenaron con un dataset de sentimientos.
    -  El proyecto solamente esta realizando una implementación o despliegue de un modelo especializado. Es una solución de ingeniería eficiente porque se está aprovechando el trabajo de expertos para integrar el análisis de sentimientos.
        
   
---

## Objetivos del proyecto

  - Generación de las diferentes categorías de sentimientos, también llamadas "Polaridad", para identificar la emoción de una persona que responde a una encuesta.
    - POS (Positivo)
    - NEU (Neutral)
    - NEG (Negativo)
  - Demostración práctica del uso de una API REST utilizando FASTAPI, el servidor UVICORN, los templates HTML JINJA2 y el uso del modelo BETO
    - FASTAPI:
      - Framework Web moderno de alto rendimiento para construir APIs en Python (3.8 o superior) basado en el estándard de "type hints" de Python
      - Es muy rápido.
      - Cuenta con Swagger UI, que te permite probar los endpoints construidos. Para usarlo solo agrega /docs y sigue las instrucciones.
      - Utiliza modelos de Pydantic para validar los datos.
    - UVICORN:
      - Servidor web ASGI (Asynchronous Server Gateway Interface) para aplicaciones Python,
      - Ideal para frameworks como FASTAPI,
      - Implementa el estandar ASGI, que permite la comunicación asincrónica entre la aplicación y el servidor, mejorando la eficiencia,
      - Soporte de WebSocket, comuniccion bidireccional en tiempo real.
    - JINJA2
      - Motor de plantillas para Python
      - Su prncipal función es tomar un archivo de texto base y combinarlo con datos dinamicos para generar archivos de salida en formatos HTML, CSV, XML
      - Los filtros permiten transformar datos
        - {{ ... }} espresiones para imprimir variables de salida
        - {% ... %} sentencias de control como bucles for o condicionales if
        - {# ... #} comentarios que no se muestran en el resultado final
      - USO DE BETO, especificamente "finiteautomata/beto-sentiment-analysis"
        - El modelo específico seleccionado es "finiteautomata/beto-sentiment-analysis" el mismo que es el resultado de un proceso de Transfer Learning / Fine-Tuning realizado por grupos especializados que le añadieron la capa de clasificación y lo entrenaron con un dataset de sentimientos.
        - El proyecto solamente esta realizando una implementación o despliegue de un modelo especializado. Es una solución de ingeniería eficiente porque se está aprovechando el trabajo de expertos para integrar el análisis de sentimientos.
  ---


## Ciclo de vida del desarrollo de un proyecto Análisis de Sentimientos

- El ciclo de vida de desarrollo (Software Development Life Cycle, SDLC) para un proyecto de análisis de sentimientos con BETO, FastAPI y Jinja2 se estructura en fases que integran el ciclo de vida de la Inteligencia Artificial con el desarrollo web moderno. 
1. Planificación y Definición del Alcance
    - Objetivo:
      - Establecer qué tipo de textos se analizarán (ej. reseñas, tweets) y definir las categorías de sentimiento esperadas (Positivo, Negativo, Neutro).
    - Requisitos:
        - Identificar la necesidad de procesamiento en español, justificando el uso de BETO como modelo optimizado para este idioma. 
2. Configuración del Entorno y Adquisición del Modelo
    - Entorno Virtual:
      - Crear un entorno aislado e instalar dependencias clave: fastapi, uvicorn, jinja2 y transformers.
    - Carga del Modelo:
      - Utilizar la librería Transformers de Hugging Face para descargar el modelo pre-entrenado finitiautomata/beto-sentiment-analysis y su tokenizador correspondiente. 
3. Desarrollo del Backend (FastAPI)
    - Lógica de Inferencia:
      - Crear una función que reciba texto, lo procese mediante el tokenizador de BERT y obtenga la predicción de la categoría de sentimiento.
    - Definición de Endpoints:
      - Configurar rutas en FastAPI para recibir solicitudes (ej. POST /predict) que devuelvan la clasificación. 
4. Desarrollo del Frontend y Plantillas (Jinja2)
    - Configuración de Templates:
      - Instanciar Jinja2Templates apuntando a un directorio de plantillas HTML.
    - Interfaz de Usuario:
      - Crear un formulario en HTML donde el usuario ingrese el texto y una página de resultados que muestre la categoría de sentimiento calculada mediante etiquetas de Jinja2. 
5. Integración y Pruebas
    - Pruebas Unitarias:
      - Verificar que el modelo clasifique correctamente frases de prueba conocidas.
    - Validación de Datos:
      - Usar los modelos de Pydantic integrados en FastAPI para asegurar que las entradas de texto sean válidas. 
6. Despliegue y Mantenimiento
    - Contenerización:
      - Opcionalmente usar Docker para empaquetar la aplicación, facilitando su ejecución en servidores o la nube.
    - Monitoreo:
      - Evaluar el rendimiento del modelo en producción y actualizar si surgen nuevas necesidades de clasificación. 

Estas fases son a menudo iterativas, donde el ajuste del modelo o la recopilación de datos adicionales pueden ser necesarios después de la evaluación o incluso del despliegue inicial.

---
