def evaluate_model(model, train_ds, val_ds):
    """Evaluate the model on train and validation datasets."""
    train_metrics = model.evaluate(train_ds, verbose=0, return_dict=True)
    val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)

    return {
        "train": train_metrics,
        "val": val_metrics,
    }


def summarize_best_validation(history):
    """Extract the best validation results from the training history."""
    best_epoch = min(
        range(len(history.history["val_loss"])),
        key=lambda i: history.history["val_loss"][i],
    ) + 1

    best_val_loss = min(history.history["val_loss"])
    best_val_accuracy = history.history["val_accuracy"][best_epoch - 1]

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
    }