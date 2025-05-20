import logging
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from .preprocessing import load_and_preprocess_data, create_preprocessor, balance_dataset
from .models import create_model_pipeline, train_model, evaluate_model, save_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    directories = ['models', 'reports', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    setup_directories()

    data_path = "data/final_project.ods"
    model_path = "models/career_level_model.joblib"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/classification_report_{timestamp}.txt"
    cm_path = f"reports/confusion_matrix_{timestamp}.png"

    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

        # Create preprocessor
        logger.info("Creating preprocessor...")
        preprocessor = create_preprocessor()

        # Balance dataset
        logger.info("Balancing dataset...")
        X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

        # Create and train model
        logger.info("Creating model pipeline...")
        model_pipeline = create_model_pipeline(preprocessor)

        logger.info("Training model...")
        model = train_model(X_train_balanced, y_train_balanced, model_pipeline)

        # Evaluate model
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)

        # Save results
        logger.info("Saving results...")
        save_model(model, model_path)

        with open(report_path, 'w') as f:
            f.write(report)

        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, model.classes_, cm_path)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Classification report saved to {report_path}")
        logger.info(f"Confusion matrix saved to {cm_path}")

        # Print best parameters
        logger.info("Best parameters:")
        for param, value in model.best_params_.items():
            logger.info(f"{param}: {value}")

        # Print best score
        logger.info(f"Best cross-validation score: {model.best_score_:.3f}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
