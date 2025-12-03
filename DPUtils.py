import matplotlib.pyplot as plt


def plot_history(history):
    """
    Toma el objeto history de Keras y grafica:
    1. Loss total (entrenamiento vs validación)
    2. Accuracy de la predicción de Acciones (action_output)
    3. Accuracy de la predicción de Relaciones de Dependencia (deprel_output)
    """
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    # Configurar el tamaño de la figura (ancho, alto)
    plt.figure(figsize=(18, 5))

    # --- GRÁFICA 1: PÉRDIDA TOTAL (LOSS) ---
    plt.subplot(1, 3, 1)
    plt.plot(epochs, hist['loss'], 'bx--', label='Training Loss')
    plt.plot(epochs, hist['val_loss'], 'rx--', label='Validation Loss')
    plt.title('Total Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # --- GRÁFICA 2: ACCURACY DE ACCIONES (Transitions) ---
    # Keras nombra las métricas como: nombreCapa_metrica
    plt.subplot(1, 3, 2)
    plt.plot(epochs, hist['action_output_accuracy'], 'bx--', label='Train Action Acc')
    plt.plot(epochs, hist['val_action_output_accuracy'], 'rx--', label='Val Action Acc')
    plt.title('Action Prediction Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # --- GRÁFICA 3: ACCURACY DE DEPRELS (Labels) ---
    plt.subplot(1, 3, 3)
    plt.plot(epochs, hist['deprel_output_accuracy'], 'bx--', label='Train Deprel Acc')
    plt.plot(epochs, hist['val_deprel_output_accuracy'], 'rx--', label='Val Deprel Acc')
    plt.title('Dependency Label Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    