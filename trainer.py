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





