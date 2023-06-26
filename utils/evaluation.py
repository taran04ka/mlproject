import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Evaluator:
    def evaluate_model(self, model, test_images, test_labels):
        y_pred = model.predict(test_images)
        y_pred = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(test_labels, axis=1)
        print(classification_report(true_labels, y_pred))
        cm = confusion_matrix(true_labels, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.show()

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()