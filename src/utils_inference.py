import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
def softmax(x):
    exp_scores = np.exp(x)
    softmax_probs = exp_scores / np.sum(exp_scores, axis=0)
    return softmax_probs
def get_accuracy_score(exp,idx_to_time_indices):
    true_classes = []
    pred_classes = []

    for i in range(len(exp.model['class_score'])):
        # Calculate the predicted class using softmax and argmax
        predicted_class = np.argmax(softmax(exp.model['class_score'][i]))
        # Get the actual class label
        true_class = np.argmax(exp.model['class_labels'][i])
        
        # Append the true and predicted classes to their respective lists
        true_classes.append(true_class)
        pred_classes.append(predicted_class)

    # Convert numerical labels to string labels
    true_classes_str = [idx_to_time_indices[cls] for cls in true_classes]
    pred_classes_str = [idx_to_time_indices[cls] for cls in pred_classes]


    # Generate the confusion matrix
    conf_matrix = confusion_matrix(true_classes_str, pred_classes_str, labels=list(idx_to_time_indices.values()))

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate and print classification metrics
    accuracy = accuracy_score(true_classes_str, pred_classes_str)
    class_report = classification_report(true_classes_str, pred_classes_str)

    print("\nClassification Report:")
    print(class_report)
    print("Accuracy:", accuracy)

def plot_loss_curves(train_loss, val_loss):
    loss_categories=7
    plot_names = [
        '{a) Combined Loss', ' (b) Reconstruction Loss', '(d) SINDy_x Loss', 
        '(e) SINDy_z Loss', '(f) SINDy Norm', ' (g) Autoencoder Norm', '(h) Classification Error'
    ]
    epochs = len(train_loss)
    x_values = np.arange(epochs) * 50

    for i in range(loss_categories):
        if plot_names[i] == 'Autoencoder Norm':
            continue  # Skip this category
        
        # Create a new figure for each plot
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, train_loss[:, i], label=f'Training Loss')
        plt.plot(x_values, val_loss[:, i], label=f'Validation Loss', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (log scale)')
        plt.title(plot_names[i])
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        # Save each figure individually
        plt.tight_layout()
        # plt.savefig(f"{plot_names[i]}_Loss_Curve.pdf", format='pdf', bbox_inches='tight')
        plt.show()
