# IA Services

Este repositorio contiene una colección de proyectos personales de **Deep Learning** y **Machine Learning**, organizados en diferentes módulos y servicios.  
El objetivo es construir, entrenar y desplegar modelos de inteligencia artificial en distintas áreas, con un enfoque reproducible y escalable.

---

## Introducción

La carpeta principal `IA-services` agrupa proyectos relacionados con aprendizaje automático y profundo.  
Cada proyecto está estructurado con:
- Datos (`datasets/`)
- Código fuente en Python (`python/`)
- Imágenes de contenedores (`container-images/`)
- Documentación y resultados

---

## Estructura del repositorio

```
IA-services
├── container-images/        # Imágenes de contenedores (Docker, Singularity, etc.)
├── datasets/                # Conjuntos de datos para entrenamiento y validación
│   └── german_credit_risk.csv
├── python/                  # Proyectos en Python
│   └── credit-scoring/      # Proyecto de scoring crediticio con PyTorch
│       ├── artifacts/
│       ├── config/
│       ├── mlruns/
│       ├── models/
│       ├── reports/
│       ├── src/
│       ├── test/
│       ├── Dockerfile
│       ├── requeriments.txt
│       └── README.md
└── README.md                # Documentación general del repositorio
```

---

## Proyectos actuales

### 1. Credit Scoring (PyTorch)
Modelo de red neuronal multicapa (MLP) entrenado para predecir el riesgo crediticio de clientes.  
Incluye:
- Configuración modular en YAML
- Entrenamiento con MLflow
- Guardado y versionado de modelos
- Visualización de métricas en `reports/`
- Despliegue con Docker

### 2. Otros proyectos (en construcción)
La idea es ir incorporando más módulos, por ejemplo:
- Procesamiento de imágenes (CNN)
- Procesamiento de lenguaje natural (RNN, Transformers)
- Series temporales y predicción de secuencias
- Experimentación con cuantización y optimización en CPU/GPU

---

## Datos

Los datasets se almacenan en `datasets/`.  
Actualmente se incluye el dataset **German Credit Risk** para el proyecto de scoring. 
Link de el repositorio: https://github.com/alicenkbaytop/German-Credit-Risk-Classification

Cada nuevo proyecto podrá añadir sus propios datasets.  

---

## Despliegue con contenedores

**(El manejo de Dockerfile aun sigue en desarrollo, por el momento los contenedores estan vacios!)**
Los contenedores se encuentran en `container-images/`.  
Cada proyecto puede generar su propia imagen con un `Dockerfile` y ser desplegado de manera independiente.  

Ejemplo (para credit scoring):

```bash
cd python/credit-scoring
docker build -t credit-scoring .
docker run -p 8080:8080 credit-scoring
```

---

## Requerimientos generales

Cada proyecto incluye un archivo `requeriments.txt` con sus dependencias específicas.  
Para un entorno general de trabajo:

```bash
pip install torch torchvision mlflow scikit-learn pandas numpy matplotlib
```

---

## Plan futuro

- Ampliar la colección de proyectos de deep learning.  
- Integrar CI/CD para pruebas automáticas.  
- Documentar con `pdoc` o `Sphinx` cada subproyecto.  
- Publicar algunos modelos en Hugging Face Hub para compartir resultados.  

---

## Licencia

Repositorio personal de investigación y práctica.  
Puedes reutilizar la estructura para tus propios proyectos de aprendizaje profundo.  
