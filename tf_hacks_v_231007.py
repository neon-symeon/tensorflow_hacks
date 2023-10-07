def plot_loss_curves(history):
    """
    Plots model performance of loss and accuracy,
    both training and validationd scores

    Args:
    history (keras.src.callbacks.History): tensorflow history object

    Returns:
    plots training/validation loss and accuracy metrics.
    """
    import matplotlib.pyplot as plt

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Utwórz siatkę 1x2 (jeden wiersz, dwie kolumny) dla wykresów
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Wykres straty
    axs[0].plot(epochs, loss, label='training')
    axs[0].plot(epochs, val_loss, label='validate')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Score')
    axs[0].legend()

    # Wykres dokładności
    axs[1].plot(epochs, accuracy, label='training')
    axs[1].plot(epochs, val_accuracy, label='validate')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('%')
    axs[1].legend()

    plt.tight_layout()  # Zapewnia odpowiednie rozmieszczenie wykresów
    plt.show()
