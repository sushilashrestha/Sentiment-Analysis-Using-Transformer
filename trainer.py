# # Importing Necessary Libraries
# import tqdm
# import os
# import torch
# import torch.nn as nn
# from activation import Softmax
# from encoder import Encoder

# # class which contains utility functions for training and evaluating our SA model.
# class Trainer():

#     def __init__(self,
#                  train_iterator,
#                  train_batch_per_epoch,
#                  test_iterator,
#                  eval_batch_per_epoch,
#                  enc_vocab_size,
#                  out_size,
#                  max_seq_len,
#                  d_model,
#                  pad_ix=0,
#                  pooling = "max",
#                  num_heads=4,
#                  expansion_factor=2,
#                  num_blocks=2,
#                  activation="relu",
#                  dropout_size=None,
#                  model_save_path=None,
#                  criterator="cel",
#                  optimizer_type="adamw"
#                  ):

#         """
#         A Utility Class for training and evaluating our sentiment analysis model.

#         Args:
#             train_iterator: Data generator for training.
#             train_batch_per_epoch (int): Number of training batches per epoch.
#             test_iterator: Data generator for evaluation.
#             eval_batch_per_epoch (int): Number of evaluation batches per epoch.
#             enc_vocab_size (int): Size of the vocabulary.
#             out_size (int): Output size of the model (Number of the Output Nodes).
#             max_seq_len (int): Maximum sequence length that input can have.
#             d_model (int): Dimension of the model.
#             pad_ix (int): Padding token index. Default is 0.
#             pooling (str): Pooling method. Default is "max".
#             num_heads (int): Number of attention heads. Default is 4.
#             expansion_factor (int): Expansion factor for feedforward layer. Default is 2.
#             num_blocks (int): Number of encoder blocks. Default is 2.
#             activation (str): Activation function. Default is "relu".
#             dropout_size (float): Dropout rate. Default is None.
#             model_save_path (str): Path to save the trained model. Default is None.
#             criterator (str): Loss Function to calculate the loss of the current batch. Default is "cel". Other Options: "bce"
#             optimizer_type (str): Optimizing Algorithm used for training. Default is "adamw". Other Options: "radam". Optimizer adjust the parameters of the model to minimize the loss function during training
#         """

#         self.train_generator = train_iterator
#         self.batch_per_epoch_train = train_batch_per_epoch
#         self.test_generator = test_iterator
#         self.batch_per_epoch_eval = eval_batch_per_epoch
#         self.enc_vocab_size = enc_vocab_size
#         self.out_size = out_size
#         self.max_seq_len = max_seq_len
#         self.d_model = d_model
#         self.padding_id = pad_ix
#         self.pooling = pooling
#         self.num_heads = num_heads
#         self.expansion_factor = expansion_factor
#         self.num_blocks = num_blocks
#         self.activation = activation
#         self.dropout_rate = dropout_size
#         self.model_save_path = model_save_path
#         self.criterator = criterator
#         self.optimizer_type = optimizer_type
#         # self.train_generator_len = len(list(self.train_generator))
#         # self.test_generator_len = len(list(self.test_generator))
#         self.softmax = Softmax(-1)

#     def prepare_model(self):

#         """
#         Initiailize and Offload the Model to GPU.

#         Returns:
#             Prepared model for Training.
#         """

#         print("Preparring the Model for Training...")
#         model = Encoder(
#             self.enc_vocab_size,
#             self.out_size,
#             self.max_seq_len,
#             padding_idx=self.padding_id,
#             pooling=self.pooling,
#             embedding_dim=self.d_model,
#             num_blocks=self.num_blocks,
#             activation=self.activation,
#             expansion_factor=self.expansion_factor,
#             num_heads=self.num_heads,
#             dropout=self.dropout_rate
#         )

#         # Getting the Current Device which we are in.
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         # Moving the Model to th Device.
#         model = model.to(self.device)

#         # if torch.__version__ >= "2":
#         #     print("Compiling Model for Speedup..")
#         #     model = torch.compile(model)

#         print("Sucessfully Prepared Model for Training.")
#         return model

#     def prepare_lossfn_optims_lr_scheduler(self, model, lr, wd, gm):

#         """
#         Prepares loss function, optimizer, and learning rate scheduler.

#         Args:
#             model: Encoder model prepared for training.
#             lr (float): Learning rate.
#             wd (float): Weight decay.
#             gm (float): Learning rate decay factor.

#         Returns:
#             tuple: Loss function, optimizer, and learning rate scheduler.
#         """

#         if self.criterator == "bce":
#             loss_fn = nn.BCE() # Binary Cross Entropy Loss for Binary Classification

#         elif self.criterator == "cel":
#             loss_fn = nn.CrossEntropyLoss() # Cross Entropy Loss for Multi class Prediction

#         if self.optimizer_type == "radam":

#             # Rectified Adam Optimizer. It address some of the limitations of the original Adam optimizer particularly it tackle the issue of "exploding" learning rates during the initial phases of training.
#             # It was Introduced in the Paper `On the variance of the adaptive learning rate and beyond`. It's an Intresting Paper Actually Go check it out if you want
#             optimizer = torch.optim.RAdam(
#                 model.parameters(),
#                 lr=lr,
#                 betas=(0.9, 0.999),
#                 eps=1e-06,
#                 weight_decay=wd,
#                 )

#         elif self.optimizer_type == 'adamw':

#             # AdamW is another variant of the Adam optimizer that addresses a specific issue related to weight decay regularization. it involves adding a penalty term to the loss function that discourages large values for the model's parameters (weights).
#             # This helps in promoting simpler models and reducing the risk of overfitting.
#             optimizer = torch.optim.AdamW(
#                 model.parameters(),
#                 lr=lr,
#                 betas=(0.9, 0.999),
#                 eps=1e-06,
#                 weight_decay=wd,
#                 )

#         # Decays the learning rate of each parameter group by gamma every epoch. It basically multiply the learning rate of the each parameter group with gamma every epoch.
#         # Thus reducing the learning rate exponentially as we get closer to the global minima.
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gm, verbose=True)

#         return loss_fn, optimizer, scheduler

#     def accuracy(self, y_pred, y_target):

#         """
#         Calculates accuracy between predicted and actual sentiment.

#         Args:
#             y_pred: Predicted labels.
#             y_target: Target labels.

#         Returns:
#             Accuracy of the given batch
#         """

#         return (torch.argmax(self.softmax(y_pred), dim=1) == y_target).float().mean()

#     def train(self, model, num_epochs=6, learning_rate=1e-3, weight_decay=1e-4, gamma=0.1):

#         """
#         Trains the Sentiment Analysis Model.

#         Args:
#             model (Encoder): Prepared model for Training.
#             num_epochs (int): Number of epochs. Default is 6.
#             learning_rate (float): Learning rate. Default is 1e-3.
#             weight_decay (float): Weight decay. Default is 1e-4.
#             gamma (float): Learning rate decay factor. Default is 0.1.

#         Returns:
#             tuple: Training losses, training accuracies, testing accuracies, path to the saved model file.
#         """

#         train_loss = []
#         train_acc = []
#         test_acc = []

#         print("Preparing loss fuction, optimizer and learning rate scheduler...")
#         learning_rate = float(learning_rate)
#         weight_decay = float(weight_decay)
#         criteration, optimizer, lr_scheduler = self.prepare_lossfn_optims_lr_scheduler(model, learning_rate, weight_decay, gamma)
#         print("Initialised Sucessfully...")

#         for epoch in range(1, num_epochs+1):

#             running_train_loss = 0.0
#             model.train(True)
#             print("\nTraining:\n")

#             # we are using tqdm's trange method to visualize the training more concisely.
#             with tqdm.trange(self.batch_per_epoch_train, unit=" batch", mininterval=0) as bar:
#                 bar.set_description(f"Epoch {epoch}/{num_epochs}")
#                 for i in bar:

#                     input_batch, label_batch, mask_batch = next(self.train_generator)
#                     input_batch, label, mask = input_batch.to(self.device), label_batch.to(self.device), mask_batch.to(self.device)

#                     # Forward Pass through the Model
#                     optimizer.zero_grad() # Zero out the Previous Gradient Calculation
#                     y_pred = model(input_batch, mask)
#                     loss = criteration(y_pred, label) # Calculating the loss for the current batch
#                     accuracy = self.accuracy(y_pred, label) # Calculating accuracy for the current batch

#                     # Store Metrics
#                     train_loss.append(loss.item())
#                     train_acc.append(accuracy.cpu().numpy().tolist())

#                     # Backward Propagation
#                     loss.backward()
#                     optimizer.step() # Adjusting all the parameters of the model using calsulated gradients and learning rate.

#                     """
#                     Behind the hoods, this is what the optimizer class is doing. You can understand underlying backward propogation by understanding the code.
#                     (Note: Diffrentiation (Backward Propagation) is not shown here.)

#                     class Optimizer:

#                         def __init__(self, params, lr):
#                             self.params = list(params)
#                             self.lr = lr

#                         def step(self):
#                             with torch.no_grad():
#                                 for p in self.params:
#                                     p -= p.grad * self.lr

#                         def zero_grad(self):
#                             for p in self.params:
#                                 p.grad.zero_()
#                     """

#                     running_train_loss += loss.item()

#                     # if not i % 1000:
#                     #     print(f"Epoch: {epoch}/{num_epochs} | Batch: {i} | Loss: {loss.item()} | Accuracy: {accuracy}")

#                     # Update the Loss and Accuracy of the previous batch with current batch's item.
#                     bar.set_postfix(
#                         loss=float(loss),
#                         acc=f"{float(accuracy)*100:.2f}%"
#                     )

#                 print(f'Epoch [{epoch}/{num_epochs}] Average Training Loss: {running_train_loss/i}')

#             running_val_loss = 0.0
#             model.eval()
#             print("\nValidating:\n")

#             with torch.no_grad():
#                 correct_prediction = 0.0
#                 total = 0.0
#                 with tqdm.trange(self.batch_per_epoch_eval, unit=" batch", mininterval=0) as bar:
#                     bar.set_description(f"Calculating Accuracy")
#                     for i in bar:

#                         input_batch, label_batch, mask_batch = next(self.test_generator)
#                         input_batch, label, mask = input_batch.to(self.device), label_batch.to(self.device), mask_batch.to(self.device)

#                         pred = model(input_batch, mask)
#                         acc = self.accuracy(pred, label)
#                         correct_prediction += (((torch.argmax(self.softmax(pred), dim=1)) == label).sum()).item()
#                         total += input_batch.size(0)
#                         test_acc.append(acc.cpu().numpy().tolist())

#                         # if not i % 1000:
#                         #     print(f"Epoch: {epoch}/{num_epochs} | Batch: {i} | Loss: {loss.item()} | Accuracy: {accuracy}")

#                         # Update the Accuracy of the previous batch with current batch's accuracy.
#                         bar.set_postfix(
#                             acc=f"{float(acc)*100:.2f}%"
#                         )

#                 print(f'Epoch [{epoch}/{num_epochs}] Validation Accuracy: {float(correct_prediction/total)*100:.2f}')

#             lr_scheduler.step() # calling learning rate scheduler after every epoch to calculate the learning rate for next epoch.

#         print("Training Finished. Saving the Model...")

#         # Saving The Trained Model
#         print("Checking whether the Given Path for Saving the Model Exist...")
#         if not os.path.isdir(os.path.dirname(self.model_save_path)):
#             print("Creating directory for to save the model as it doesn't exist")
#             os.mkdir(os.path.dirname(self.model_save_path))
#             self.model_save_path = os.path.join(os.path.dirname(self.model_save_path),'/imd-sa.bin')
#             print("Saving the Model at: ", self.model_save_path)
#             torch.save(model.state_dict(), self.model_save_path)

#         else:
#             print("Saving the Model at: ", self.model_save_path)
#             torch.save(model, self.model_save_path)


#         return train_loss, train_acc, test_acc, self.model_save_path
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from activation import Softmax
from encoder import Encoder
from helpers import plot_confusion_matrix, calculate_metrics,plot_roc_curve
from sklearn.metrics import confusion_matrix

class Trainer():
    def __init__(self,
                 train_iterator,
                 train_batch_per_epoch,
                 test_iterator,
                 eval_batch_per_epoch,
                 enc_vocab_size,
                 out_size,
                 max_seq_len,
                 d_model,
                 pad_ix=0,
                 pooling="max",
                 num_heads=4,
                 expansion_factor=2,
                 num_blocks=2,
                 activation="relu",
                 dropout_size=None,
                 model_save_path=None,
                 criterator="cel",
                 optimizer_type="adamw"
                 ):
        self.train_generator = train_iterator
        self.batch_per_epoch_train = train_batch_per_epoch
        self.test_generator = test_iterator
        self.batch_per_epoch_eval = eval_batch_per_epoch
        self.enc_vocab_size = enc_vocab_size
        self.out_size = out_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.padding_id = pad_ix
        self.pooling = pooling
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        self.num_blocks = num_blocks
        self.activation = activation
        self.dropout_rate = dropout_size
        self.model_save_path = model_save_path
        self.criterator = criterator
        self.optimizer_type = optimizer_type
        self.softmax = Softmax(-1)

    def prepare_model(self):
        print("Preparing the Model for Training...")
        model = Encoder(
            self.enc_vocab_size,
            self.out_size,
            self.max_seq_len,
            padding_idx=self.padding_id,
            pooling=self.pooling,
            embedding_dim=self.d_model,
            num_blocks=self.num_blocks,
            activation=self.activation,
            expansion_factor=self.expansion_factor,
            num_heads=self.num_heads,
            dropout=self.dropout_rate
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        print("Successfully Prepared Model for Training.")
        return model

    def prepare_lossfn_optims_lr_scheduler(self, model, lr, wd, gm):
        if self.criterator == "bce":
            loss_fn = nn.BCE()
        elif self.criterator == "cel":
            loss_fn = nn.CrossEntropyLoss()

        if self.optimizer_type == "radam":
            optimizer = torch.optim.RAdam(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-06,
                weight_decay=wd,
            )
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-06,
                weight_decay=wd,
            )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gm, verbose=True)

        return loss_fn, optimizer, scheduler

    def accuracy(self, y_pred, y_target):
        return (torch.argmax(self.softmax(y_pred), dim=1) == y_target).float().mean()

    # def train(self, model, num_epochs=6, learning_rate=1e-3, weight_decay=1e-4, gamma=0.1):
    #     train_loss = []
    #     train_acc = []
    #     test_loss = []
    #     test_acc = []

    #     print("Preparing loss function, optimizer and learning rate scheduler...")
    #     learning_rate = float(learning_rate)
    #     weight_decay = float(weight_decay)
    #     criteration, optimizer, lr_scheduler = self.prepare_lossfn_optims_lr_scheduler(model, learning_rate, weight_decay, gamma)
    #     print("Initialized Successfully...")

    #     for epoch in range(1, num_epochs+1):
    #         running_train_loss = 0.0
    #         correct_prediction = 0.0
    #         total_samples = 0.0

    #         model.train(True)
    #         print("\nTraining:\n")

    #         for i in range(self.batch_per_epoch_train):
    #             input_batch, label_batch, mask_batch = next(self.train_generator)
    #             input_batch, label, mask = input_batch.to(self.device), label_batch.to(self.device), mask_batch.to(self.device)

    #             optimizer.zero_grad()
    #             y_pred = model(input_batch, mask)
    #             loss = criteration(y_pred, label)
    #             accuracy = self.accuracy(y_pred, label)

        

    #             running_train_loss += loss.item()

    #             correct_prediction += ((torch.argmax(self.softmax(y_pred), dim=1) == label).sum()).item()
    #             total_samples += label.size(0)

    #             loss.backward()
    #             optimizer.step()

    #         epoch_train_loss = running_train_loss / self.batch_per_epoch_train
    #         train_loss.append(epoch_train_loss)

    #         epoch_train_accuracy = correct_prediction / total_samples
    #         train_acc.append(epoch_train_accuracy)

    #         print(f'Epoch [{epoch}/{num_epochs}] Average Training Loss: {epoch_train_loss}')
    #         print(f'Epoch [{epoch}/{num_epochs}] Training Accuracy: {epoch_train_accuracy * 100:.2f}%')

    #         running_val_loss = 0.0
    #         running_val_acc = 0.0
    #         model.eval()
    #         print("\nValidating:\n")

    #         with torch.no_grad():
    #             correct_prediction = 0.0
    #             total = 0.0
    #             predicted_labels= []
    #             true_labels = []
                
    #             for i in range(self.batch_per_epoch_eval):
    #                 input_batch, label_batch, mask_batch = next(self.test_generator)
    #                 input_batch, label, mask = input_batch.to(self.device), label_batch.to(self.device), mask_batch.to(self.device)

    #                 pred = model(input_batch, mask)
    #                 loss = criteration(pred, label)
    #                 acc = self.accuracy(pred, label)

    #                 running_val_loss += loss.item()

    #                 pred_labels_batch = torch.argmax(pred, dim=1)
    #                 predicted_labels.extend(pred_labels_batch.tolist())
    #                 true_labels.extend(label.tolist())

    #                 correct_prediction += ((torch.argmax(self.softmax(pred), dim=1) == label).sum()).item()
    #                 total += label.size(0)

    #                 running_val_acc += acc.item()

    #             epoch_test_loss = running_val_loss / self.batch_per_epoch_eval
    #             test_loss.append(epoch_test_loss)

    #             epoch_test_accuracy = correct_prediction / total
    #             test_acc.append(epoch_test_accuracy)

    #             print(f'Epoch [{epoch}/{num_epochs}] Validation Loss: {epoch_test_loss}')
    #             print(f'Epoch [{epoch}/{num_epochs}] Validation Accuracy: {epoch_test_accuracy * 100:.2f}%')

    #         lr_scheduler.step()

    #     print("Training Finished. Saving the Model...")

    #     if not os.path.isdir(os.path.dirname(self.model_save_path)):
    #         print("Creating directory for to save the model as it doesn't exist")
    #         os.mkdir(os.path.dirname(self.model_save_path))
    #         self.model_save_path = os.path.join(os.path.dirname(self.model_save_path), '/imd-sa.bin')
    #         print("Saving the Model at: ", self.model_save_path)
    #         torch.save(model, self.model_save_path)

    #     else:
    #         print("Saving the Model at: ", self.model_save_path)
    #         torch.save(model, self.model_save_path)
    #     #compute the confusion matrix 
            
    #     cm = confusion_matrix(true_labels, predicted_labels)
    #     plot_confusion_matrix(cm)
    #     return train_loss, train_acc, test_loss, test_acc, self.model_save_path

    def train(self, model, num_epochs=6, learning_rate=1e-3, weight_decay=1e-4, gamma=0.1):
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        print("Preparing loss function, optimizer and learning rate scheduler...")
        learning_rate = float(learning_rate)
        weight_decay = float(weight_decay)
        criterion, optimizer, lr_scheduler = self.prepare_lossfn_optims_lr_scheduler(model, learning_rate, weight_decay, gamma)
        print("Initialized Successfully...")

        for epoch in range(1, num_epochs+1):
            running_train_loss = 0.0
            correct_prediction = 0.0
            total_samples = 0.0

            model.train(True)
            print("\nTraining:\n")

            progress_bar_train = tqdm(range(self.batch_per_epoch_train))
            for i in progress_bar_train:
                input_batch, label_batch, mask_batch = next(self.train_generator)
                input_batch, label, mask = input_batch.to(self.device), label_batch.to(self.device), mask_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = model(input_batch, mask)
                loss = criterion(y_pred, label)
                accuracy = self.accuracy(y_pred, label)

                running_train_loss += loss.item()

                correct_prediction += ((torch.argmax(self.softmax(y_pred), dim=1) == label).sum()).item()
                total_samples += label.size(0)

                loss.backward()
                optimizer.step()

                progress_bar_train.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2f}%")

            epoch_train_loss = running_train_loss / self.batch_per_epoch_train
            train_loss.append(epoch_train_loss)

            epoch_train_accuracy = correct_prediction / total_samples
            train_acc.append(epoch_train_accuracy)

            print(f'Epoch [{epoch}/{num_epochs}] Average Training Loss: {epoch_train_loss}')
            print(f'Epoch [{epoch}/{num_epochs}] Training Accuracy: {epoch_train_accuracy * 100:.2f}%')

            running_val_loss = 0.0
            running_val_acc = 0.0
            model.eval()
            print("\nValidating:\n")

            with torch.no_grad():
                correct_prediction = 0.0
                total = 0.0
                predicted_labels= []
                true_labels = []
                
                progress_bar_val = tqdm(range(self.batch_per_epoch_eval))
                for i in progress_bar_val:
                    input_batch, label_batch, mask_batch = next(self.test_generator)
                    input_batch, label, mask = input_batch.to(self.device), label_batch.to(self.device), mask_batch.to(self.device)

                    pred = model(input_batch, mask)
                    loss = criterion(pred, label)
                    acc = self.accuracy(pred, label)

                    running_val_loss += loss.item()

                    pred_labels_batch = torch.argmax(pred, dim=1)
                    predicted_labels.extend(pred_labels_batch.tolist())
                    true_labels.extend(label.tolist())

                    correct_prediction += ((torch.argmax(self.softmax(pred), dim=1) == label).sum()).item()
                    total += label.size(0)

                    running_val_acc += acc.item()

                    progress_bar_val.set_description(f"Epoch {epoch}/{num_epochs}, Validation Loss: {loss.item():.4f}, Accuracy: {acc.item():.2f}%")

                epoch_test_loss = running_val_loss / self.batch_per_epoch_eval
                test_loss.append(epoch_test_loss)

                epoch_test_accuracy = correct_prediction / total
                test_acc.append(epoch_test_accuracy)

                print(f'Epoch [{epoch}/{num_epochs}] Validation Loss: {epoch_test_loss}')
                print(f'Epoch [{epoch}/{num_epochs}] Validation Accuracy: {epoch_test_accuracy * 100:.2f}%')

            lr_scheduler.step()

        print("Training Finished. Saving the Model...")

        if not os.path.isdir(os.path.dirname(self.model_save_path)):
            print("Creating directory for to save the model as it doesn't exist")
            os.mkdir(os.path.dirname(self.model_save_path))
            self.model_save_path = os.path.join(os.path.dirname(self.model_save_path), '/imd-sa.bin')
            print("Saving the Model at: ", self.model_save_path)
            torch.save(model, self.model_save_path)

        else:
            print("Saving the Model at: ", self.model_save_path)
            torch.save(model, self.model_save_path)
            
        # Compute the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        plot_confusion_matrix(cm)
        # Extracting TP, TN, FP, FN from confusion matrix
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]

        print("True Positives:", TP)
        print("True Negatives:", TN)
        print("False Positives:", FP)
        print("False Negatives:", FN)

        calculate_metrics(TP,TN, FP, FN)
        plot_roc_curve(true_labels,predicted_labels)
        

        return train_loss, train_acc, test_loss, test_acc, self.model_save_path





