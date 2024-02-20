from torchinfo import summary
from prettytable import PrettyTable
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve,auc


# Print a Comprehensive Summary of the Model, Modules, Submodules, Parameter Counts
def model_summary(model, generator):
    review_batch, label, mask_batch = next(generator)
    print(summary(model, input_data=[review_batch.to("cuda:0"), mask_batch.to("cuda:0")]))


# Utility function to print the Modules, SubModules and their Corresponding trainable parmeters in a Clean Table Structure
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# plot the training loss and training, testing accuracy
#def plot_metrics(num_epochs, batch_per_epoch_train,batch_per_epoch_test,train_loss, train_acc, test_acc):

#     # X axis Epoch
#     train_epochs = list(range(1, (batch_per_epoch_train * num_epochs) + 1))
#     test_epochs = list(range(1, (batch_per_epoch_test * num_epochs) + 1))
    
#     # Create subplots for train loss and accuracy
#     plt.figure(figsize=(24, 3))
    
#     # Plot Train Loss
#     plt.plot(train_epochs, train_loss, label='Train Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss')
#     plt.legend()
    
#     # Plot Train Accuracy
#     plt.figure(figsize=(24, 3))
#     # plt.subplot(1, 2, 2)
#     plt.plot(train_epochs, train_acc, label='Train Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Training Accuracy')
#     plt.legend()
    
#     # Plot Test Accuracy
#     plt.figure(figsize=(24, 3))
#     plt.plot(test_epochs, test_acc, label='Test Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Testing Accuracy')
#     plt.legend()
    
#     # Adjust layout
#     #plt.tight_layout()
    
#     # Show the plots
#     plt.show()

#in single frame
# def plot_metrics(num_epochs, batch_per_epoch_train, batch_per_epoch_test, train_loss, train_acc, test_acc):
#     # Calculate epochs
#     train_epochs = list(range(1, (batch_per_epoch_train * num_epochs) + 1))
#     test_epochs = list(range(1, (batch_per_epoch_test * num_epochs) + 1))
    
#     # Create subplots
#     fig, axes = plt.subplots(3, 1, figsize=(12,12))
    
#     # Plot Train Loss
#     axes[0].plot(num_epochs, train_loss, label='Train Loss')
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylabel('Loss')
#     axes[0].set_title('Training Loss')
#     axes[0].legend()
    
#     # Plot Train Accuracy
#     axes[1].plot(num_epochs, train_acc, label='Train Accuracy')
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('Accuracy')
#     axes[1].set_title('Training Accuracy')
#     axes[1].legend()
    
#     # Plot Test Accuracy
#     axes[1].plot(num_epochs, test_acc, label='Test Accuracy')
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('Accuracy')
#     axes[1].set_title('Testing Accuracy')
#     axes[1].legend()
    
#     # Adjust layout
#     plt.subplots_adjust(hspace=0.75)
#     #plt.tight_layout()
    
#     # Show the plots
#     plt.show()




def plot_metrics(train_loss, train_acc, test_loss, test_acc):
    epochs = list(range(1, len(train_acc) + 1))
   

    plt.figure(figsize=(12, 5))

    # Plot Training and Testing Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b', label='Training acc')
    plt.plot(epochs, test_acc, 'r', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Training and Testing Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, test_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()




def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def calculate_metrics(TP,TN,FP,FN):
    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = TP / (TP + FP)
    print("Precision:", precision)

    # Calculate recall
    recall = TP / (TP + FN)
    print("Recall:", recall)

    # Calculate F1 score
    f1 = (2 * precision * recall) / (precision + recall)
    print("F1 Score:", f1)
    

def plot_roc_curve(y_true, y_score):
    """
    Function to plot the ROC curve.
    
    Parameters:
        y_true: array-like, true binary labels.
        y_score: array-like, predicted probabilities or decision function scores.
    
    Returns:
        None (plots ROC curve)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()