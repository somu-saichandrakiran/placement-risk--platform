# ml/save_model.py
import pickle
from ml.features import load_and_clean
from ml.pipeline import build_placement_pipeline
from sklearn.model_selection import train_test_split
from loguru import logger

def save_model(
    data_path: str = "data/raw/placements_raw.csv",
    output_path: str = "model.pkl"
):
    X, y, _ = load_and_clean(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info("Training final model...")
    pipeline = build_placement_pipeline()
    pipeline.fit(X_train, y_train)

    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info(f"Model saved to {output_path}")
    print(f"Done. Model saved to {output_path}")

if __name__ == "__main__":
    save_model()