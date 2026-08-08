# streamlit-ML

Proyecto demostrativo para crear una plataforma de clasificación de dígitos con Streamlit Cloud.

## Objetivo
Permitir al usuario dibujar un número en un lienzo y clasificarlo con 3 modelos de ML entrenados en el dataset MNIST.

## Contenidos
- `main.py`: interfaz Streamlit con lienzo de dibujo y clasificación.
- `train_models.py`: script para descargar MNIST, entrenar los modelos y guardarlos en `models/`.
- `models/`: carpeta con los modelos entrenados serializados.

## Modelos incluidos
- Logistic Regression
- Random Forest
- K-Nearest Neighbors

## Instalación
1. Instala dependencias:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

2. Si prefieres usar pyproject.toml:

```bash
python -m pip install .
```

## Entrenar los modelos
Ejecuta:

```bash
python train_models.py
```

Esto descargará MNIST, entrenará los tres modelos y guardará los archivos `*.joblib` en `models/`.

## Ejecutar la aplicación

```bash
streamlit run main.py
```

## Uso
1. Dibuja un número en el lienzo.
2. Pulsa `Clasificar`.
3. Visualiza la predicción y la confianza de cada modelo.

## Nota
Asegúrate de ejecutar `python train_models.py` antes de abrir la app para que los modelos estén disponibles.
