from torchinfo import summary
from prettytable import PrettyTable
from matplotlib import pyplot as plt


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
def plot_metrics(num_epochs, batch_per_epoch_train,batch_per_epoch_test, train_loss, train_acc, test_acc):

    # X axis Epoch
    train_epochs = list(range(1, (batch_per_epoch_train * num_epochs) + 1))
    test_epochs = list(range(1, (batch_per_epoch_train * num_epochs) + 1))
    
    # Create subplots for train loss and accuracy
    plt.figure(figsize=(24, 3))
    
    # Plot Train Loss
    plt.plot(train_epochs, train_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Plot Train Accuracy
    plt.figure(figsize=(24, 3))
    # plt.subplot(1, 2, 2)
    plt.plot(train_epochs, train_acc, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    # Plot Test Accuracy
    plt.figure(figsize=(24, 3))
    plt.plot(test_epochs, test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy')
    plt.legend()
    
    # Adjust layout
    #plt.tight_layout()
    
    # Show the plots
    plt.show()