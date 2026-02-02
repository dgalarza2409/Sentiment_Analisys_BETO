# Teoría y desarrollo práctico de un proyecto de Análisis de sentimientos utilizando FastApi, Uvicorn, Jinja2 templates 

### Un modelo de IA para utilizar de una manera eficiente los LLM (Large Language Model) previamente entrenados con inmensos datasets, y de esta manera evitar el alto costo de un desarrollo desde cero.

Se escogio se modelo BETO por las siguientes razones (NOTA. El texto a continuación fue generado por la IA) :

  - Es la versión en **español** del modelo BERT, desarrollada por el Centro de Investigación de la Web de la Universidad de Chile.
    - Entrenamiento nativo en Español
    - Costo: Totalmente gratuito para descargar y usar.
    - Licencia: Utiliza generalmente una licencia CC BY 4.0, lo que permite su uso y adaptación citando a los autores.
    - Dónde encontrarlo: Está disponible en plataformas como Hugging Face para que desarrolladores lo integren en sus proyectos.  
  - Captura de Matices y Contexto
    - BETO utiliza una técnica llamada Whole Word Masking (WWM), que le obliga a predecir palabras completas en lugar de solo fragmentos. En el análisis de sentimientos, esto es vital para identificar: 
      - Sarcasmo e ironía: Entiende el contexto de la frase completa para detectar si una palabra positiva se usa con intención negativa.
      - Negaciones: Procesa la relación entre palabras (como "no estoy feliz") de forma bidireccional, evitando clasificaciones erróneas comunes en modelos más simples.
  - Rendimiento Superior (Benchmarks)
    - En diversas competencias de procesamiento de lenguaje natural (NLP) aplicadas al español, BETO ha demostrado superar al BERT original:
      - Precisión: Se han registrado casos donde BETO alcanza un 90.95% de precisión frente al 45.73% de modelos multilingües en tareas de detección de desinformación y sentimientos.
      - Líder en tareas específicas: Sigue siendo el estándar a batir en tareas de etiquetado de partes de la oración (POS) y reconocimiento de entidades (NER), fundamentales para desglosar "quién siente qué"
  - Transfer Learning Eficiente
    - Gracias a que ya posee una "base de conocimiento" del idioma, solo se necesita realizar un ajuste fino (fine-tuning) con un conjunto de datos pequeño de los sentimientos específicos (ej. reseñas de productos, comentarios de Twitter) para obtener resultados profesionales sin necesidad de un entrenamiento costoso.
        
   
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
  ---

## Redes convolucionales

Convolutional Neural Networks son sistemas de inteligencia artificial basados en redes neurales de varias capas que pueden identificar,reconocer y clasificar objetos, asi como detectar y segmentar objetos dentro de la imagen.

La operación de convolución combina dos funciones o datasets (la imagen y un filtro o "kernel") para producir un mapa de características ("features map").

Las capas convolucionales realizan una operación matemática llamada convolución, una especie de multiplicación matricial especializada, sobre los datos de entrada. La operación de convolución ayuda a preservar la relación espacial entre píxeles aprendiendo las características de la imagen usando pequeños cuadrados de datos de entrada.

El procesamiento de imágenes en una Red Neuronal Convolucional (CNN) funciona mediante una serie de capas que aprenden y extraen características automáticamente de los datos visuales, desde rasgos básicos como bordes hasta patrones complejos, sin necesidad de ingeniería manual de características.

El proceso paso a paso es el siguiente:
1. Capa de Entrada (Input Layer):
    - La imagen digital (que es una matriz de píxeles) se introduce en la red.
2. Capas Convolucionales (Convolutional Layers):
    - Esta es la capa principal y donde ocurre la mayor parte del procesamiento.
    - Se aplican filtros (también llamados núcleos o kernels) a la imagen. Cada filtro se desliza sobre la imagen (operación de convolución), detectando patrones locales específicos, como bordes, texturas o esquinas.
    - La salida de cada filtro es un mapa de características (feature map), que indica dónde se encontró ese patrón específico en la imagen.
    - A menudo, después de la convolución, se aplica una función de activación no lineal (como ReLU, Rectified Linear Unit) para introducir no linealidad en el modelo.
3. Capas de Agrupamiento (Pooling Layers):
    - Estas capas se intercalan entre las capas convolucionales.
    - Su función es reducir la dimensionalidad de los mapas de características, lo que disminuye la cantidad de cálculos y ayuda a evitar el sobreajuste (overfitting).
    - El tipo más común es el Max Pooling, que toma el valor máximo de un área pequeña del mapa de características, conservando la información más relevante.
4. Capas Totalmente Conectadas (Fully Connected Layers):
    - Después de varias combinaciones de capas convolucionales y de agrupamiento, los datos se "aplanan" (flatten) en un vector unidimensional.
    - Este vector se pasa a una red neuronal tradicional (totalmente conectada), que utiliza las características extraídas para realizar la tarea final de clasificación o detección.
    - La capa de salida proporciona el resultado final, por ejemplo, la probabilidad de que la imagen pertenezca a una clase determinada (un gato, un perro, un coche, etc.).

En resumen, las primeras capas de la CNN identifican elementos básicos y, a medida que la información pasa por capas más profundas, la red aprende a reconocer características cada vez más complejas y abstractas, hasta poder clasificar la imagen completa.

---
## Open CV

OpenCV es una biblioteca de código abierto para visión artificial que proporciona herramientas y algoritmos para procesamiento de imágenes, análisis de video y aprendizaje automático en tiempo real. Se utiliza para tareas como detección de rostros, reconocimiento de objetos y conducción autónoma, y es compatible con varios lenguajes como C++, Python y Java.
  - Características principales:
    - Código abierto y gratuito:
      - Publicada bajo la licencia BSD, lo que permite su uso tanto para fines comerciales como de investigación.
    - Multiplataforma:
      - Funciona en sistemas operativos como Linux, Windows, Mac OS y Android.
    - Multilenguaje:
      - Soporta múltiples lenguajes de programación, con C++ y Python como los más comunes, pero también Java y MATLAB.
    - Algoritmos optimizados:
      - Incluye más de 2.500 algoritmos optimizados para tareas de visión por computadora y aprendizaje automático en tiempo real.
    - Amplia gama de aplicaciones:
      - Se utiliza para proyectos como realidad aumentada, robótica, seguridad, análisis de movimiento y más.

¿Cómo funciona?

  - ***Un flujo de trabajo típico en OpenCV implica cargar una imagen o video, aplicar transformaciones (como filtrado o detección de características) y luego mostrar o guardar el resultado. Es una herramienta versátil que puede trabajar por sí sola o en combinación con otros marcos de aprendizaje profundo.***
---
## Ciclo de vida del desarrollo de un proyecto de Computer Vision

El ciclo de vida del desarrollo de un proyecto de Computer Vision es un proceso iterativo que adapta las fases tradicionales del desarrollo de software y del machine learning para abordar las particularidades de los datos visuales.
Las fases principales son:
  1. Definición y Planificación del Alcance:
      - Identificar el problema de negocio y el objetivo específico que la visión por computadora resolverá (por ejemplo, detección de defectos en una línea de producción, reconocimiento facial, etc.).
      - Establecer los requisitos del sistema, incluyendo la precisión esperada, la velocidad de procesamiento y las limitaciones de hardware/software.
      - Planificar la recopilación de datos, el presupuesto y el cronograma del proyecto.
  2. Recopilación y Preparación de Datos:
      - Adquisición de imágenes/videos: Recolectar un conjunto de datos visuales relevante y representativo del entorno real.
      - Limpieza y preprocesamiento: Normalizar, redimensionar y limpiar los datos para garantizar su calidad y uniformidad.
      - Anotación/Etiquetado: Etiquetar manualmente los datos (por ejemplo, delimitar objetos con cuadros, segmentar regiones) para que el modelo de aprendizaje supervisado pueda aprender.
      - Aumento de los datos: Aumentar artificialmente la variabilidad del conjunto de datos de entrenamiento existente para mejorar la robustez del modelo y reducir el sobreajuste (overfitting). Esto implica diseñar las transformaciones (rotaciones, recortes, cambios de brillo, etc.) que se aplicarán a las imágenes originales. Para CNN, la combinación de aumento de datos (Data Augmentation) para la clase minoritaria y el uso de ponderación de clases (Class Weights) suele ser la estrategia más robusta y recomendable, para reducir el desbalance de clases.
  3. Desarrollo y Entrenamiento del Modelo:
      - Diseño del modelo: Seleccionar la arquitectura de red neuronal o algoritmo de visión artificial adecuado para la tarea (por ejemplo, CNNs, R-CNN, YOLO, etc.).
      - Entrenamiento: Entrenar el modelo utilizando el conjunto de datos etiquetado. Esto a menudo implica el uso de técnicas como el aprendizaje por transferencia (transfer learning) si se dispone de datos limitados.
      - Validación y ajuste: Evaluar el rendimiento del modelo en un conjunto de datos de validación y ajustar los hiperparámetros para optimizar los resultados.
  4. Evaluación del Modelo:
      - Probar el modelo final con un conjunto de datos de prueba previamente no visto para medir su rendimiento real y verificar si cumple con los objetivos del proyecto.
      - Analizar métricas clave como precisión, recall, IoU (Intersección sobre Unión), y velocidad de inferencia.
  5. Implementación (Despliegue):
      - Integrar el modelo en el entorno de producción (servidor local, nube, dispositivo edge, etc.).
      - Asegurar la funcionalidad dentro del sistema de destino, que puede ser una aplicación de software, un dispositivo IoT o parte de un flujo de trabajo más amplio.
  6. Supervisión y Mantenimiento:
      - Monitorizar continuamente el rendimiento del modelo en tiempo real para detectar la desviación del modelo (model drift) o fallos.
      - Recopilar nuevos datos y reentrenar el modelo periódicamente para mantener su precisión y relevancia a lo largo del tiempo.
      - Realizar mantenimiento continuo y solucionar problemas operativos.

Estas fases son a menudo iterativas, donde el ajuste del modelo o la recopilación de datos adicionales pueden ser necesarios después de la evaluación o incluso del despliegue inicial.

---
