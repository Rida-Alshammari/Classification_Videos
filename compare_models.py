import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from train import get_validation_generator


def compare_models(old_model_path, new_model_path):
    val_gen = get_validation_generator()

    old_model = tf.keras.models.load_model(old_model_path)
    new_model = tf.keras.models.load_model(new_model_path)

    old_results = old_model.evaluate(val_gen)
    new_results = new_model.evaluate(val_gen)

    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall']
    comparison = pd.DataFrame({
        'Metric': metrics,
        'Old Model': old_results[0:4],
        'New Model': new_results[0:4]
    })

    plt.figure(figsize=(10, 6))
    plt.bar(comparison['Metric'], comparison['Old Model'],
            width=0.4, label='Old Model')
    plt.bar(comparison['Metric'], comparison['New Model'],
            width=0.4, label='New Model')
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('./images/model_comparison.png')
    plt.show()

    return comparison


if __name__ == '__main__':
    comparison_table = compare_models(
        './model/model_old_30.keras',
        './model/model_new_50.keras'
    )
    print(comparison_table)
