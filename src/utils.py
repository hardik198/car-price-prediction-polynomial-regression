from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_featured_data():
    """
    Load the featured dataset.
    """

    data_path = (
        PROJECT_ROOT /
        "data" /
        "processed" /
        "featured_data.csv"
    )

    return pd.read_csv(data_path)


def create_models_folder():
    """
    Create models folder.
    """

    models = PROJECT_ROOT / "models"

    models.mkdir(exist_ok=True)

    return models


def create_images_folder():
    """
    Create images folder.
    """

    images = PROJECT_ROOT / "images"

    images.mkdir(exist_ok=True)

    return images