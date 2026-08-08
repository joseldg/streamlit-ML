import joblib
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "K-Nearest Neighbors": MODEL_DIR / "k-nearest_neighbors.joblib",
}


def load_model(path: Path) -> Any:
    return joblib.load(path)


def preprocess_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    image = image.convert("L")
    arr = np.array(image)
    arr = 255 - arr
    arr = (arr > 50).astype(np.uint8) * 255

    coords = np.column_stack(np.where(arr > 0))
    if coords.size:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        arr = arr[y_min : y_max + 1, x_min : x_max + 1]

    cropped = Image.fromarray(arr)
    width, height = cropped.size
    if width == 0 or height == 0:
        resized = cropped.resize((28, 28), Image.Resampling.LANCZOS)
    else:
        scale = 20.0 / max(width, height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = cropped.resize(new_size, Image.Resampling.LANCZOS)

    final_image = Image.new("L", (28, 28), color=0)
    left = (28 - resized.width) // 2
    top = (28 - resized.height) // 2
    final_image.paste(resized, (left, top))

    final_arr = np.asarray(final_image).astype(np.float32) / 255.0
    return final_arr.reshape(1, -1), final_image


def generate_example_digit() -> Image.Image:
    image = Image.new("L", (28, 28), color=255)
    draw = ImageDraw.Draw(image)
    draw.line([(4, 4), (24, 4)], fill=0, width=5)
    draw.line([(20, 4), (8, 24)], fill=0, width=5)
    draw.line([(8, 24), (24, 24)], fill=0, width=5)
    return image


def classify_and_show(arr: np.ndarray) -> None:
    st.subheader("Resultados de los modelos")
    for name, model_path in MODEL_FILES.items():
        if not model_path.exists():
            st.error(f"Modelo no encontrado: {model_path.name}")
            continue

        model = load_model(model_path)
        proba = model.predict_proba(arr)[0]
        pred = model.classes_[int(np.argmax(proba))]
        confidence = float(np.max(proba) * 100)
        st.write(f"**{name}**: {pred} — confianza {confidence:.1f}%")


def render_ui() -> None:
    st.set_page_config(page_title="Digit Classifier", layout="centered")
    st.title("Clasificador de dígitos con Streamlit")
    st.write(
        "Dibuja un número en el lienzo y pulsa el botón para ver la predicción de tres modelos ML entrenados en MNIST."
    )

    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        background_color="#ffffff",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    missing_models = [name for name, path in MODEL_FILES.items() if not path.exists()]
    if missing_models:
        st.warning(
            "No se han encontrado los modelos entrenados. Ejecuta `python train_models.py` para crearlos."
        )

    st.write("---")
    st.write("**Instrucciones:** dibuja un dígito en el lienzo, luego pulsa 'Clasificar'.")
    st.write("También puedes usar el botón de ejemplo automático para clasificar un dígito de prueba.")

    if st.button("Ejemplo automático"):
        example = generate_example_digit()
        st.image(example.resize((280, 280), Image.Resampling.NEAREST), caption="Ejemplo automático de dígito", use_column_width=False)
        with st.spinner("Clasificando ejemplo automático..."):
            arr, preview = preprocess_image(example)
            st.image(preview.resize((280, 280), Image.Resampling.NEAREST), caption="Imagen 28x28 usada para clasificación", use_column_width=False)
            classify_and_show(arr)

    if canvas_result.image_data is not None and st.button("Clasificar"):
        image = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
        arr, preview = preprocess_image(image)
        st.image(preview.resize((280, 280), Image.Resampling.NEAREST), caption="Imagen 28x28 usada para clasificación", use_column_width=False)
        with st.spinner("Clasificando tu dibujo..."):
            classify_and_show(arr)

    if canvas_result.image_data is None:
        st.info("Dibuja un número y pulsa 'Clasificar' para ver los resultados.")


if __name__ == "__main__":
    render_ui()