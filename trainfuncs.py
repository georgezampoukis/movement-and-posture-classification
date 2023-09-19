import time
import numpy as np
import torch
import torch.nn as nn
import colorama
from colorama import Fore
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryAccuracy, BinaryCohenKappa, BinaryJaccardIndex, BinaryF1Score, BinaryPrecision, BinaryRecall





"""

---------------------------------- TRAINING & EVALUATION ----------------------------------


"""




def train(model, train_loader, valid_loader, epochs, optimizer, device, model_name, learning_rates, es_patience=10, use_lr_schedule=True, produce_plots=False):
    colorama.init(autoreset=True)

    best_valid_loss = float("inf")
    counter = 0

    per_epoch_train_acc = np.zeros(0, dtype=np.float32)
    per_epoch_train_loss = np.zeros(0, dtype=np.float32)
    per_epoch_valid_acc = np.zeros(0, dtype=np.float32)
    per_epoch_valid_loss = np.zeros(0, dtype=np.float32)

    # Load Dataset In RAM
    train_loader.dataset.LoadInMemory()
    valid_loader.dataset.LoadInMemory()

    bce_loss = nn.BCELoss()
    accuracy = BinaryAccuracy().to(device)

    if use_lr_schedule and es_patience > 0:
        learning_rate = yield_learning_rate(learning_rates)
        start_rate = next(learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = start_rate

    print(Fore.GREEN + "\n------------------------ Training Begins ------------------------\n")

    for epoch in range(epochs):
        start_time = time.time()

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        valid_epoch_loss = 0.0
        valid_epoch_acc = 0.0

        index = 1

        # Train Phase on Train Data
        model.train()
        for train_data in train_loader:
            train_img = train_data[0].to(device, dtype=torch.float32)
            train_label = train_data[1].to(device, dtype=torch.float32)

            optimizer.zero_grad()

            train_prediction = model(train_img)
            train_loss = bce_loss(train_prediction, train_label)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_acc = accuracy(train_prediction.detach(), train_label.detach())
            train_epoch_acc += train_acc.cpu().item()

            print(f"Epoch: {epoch + 1} | Step: {index} / {len(train_loader)} | BCE Loss: {(train_epoch_loss / index):.5f} | Acc: {train_epoch_acc / index:.5f}", end='\r')
            index += 1

        print("\n")

        # Evaluation Phase on Validation Data
        model.eval()
        with torch.no_grad():
            for valid_data in valid_loader:
                valid_img = valid_data[0].to(device, dtype=torch.float32)
                valid_label = valid_data[1].to(device, dtype=torch.float32)

                valid_prediction = model(valid_img)
                valid_loss = bce_loss(valid_prediction, valid_label)
                valid_epoch_loss += valid_loss.item()
                valid_acc = accuracy(valid_prediction, valid_label)
                valid_epoch_acc += valid_acc.cpu().item()

        # Calculate Epoch Average Metrics           
        train_epoch_loss /= len(train_loader)
        train_epoch_acc /= len(train_loader)
        valid_epoch_loss /= len(valid_loader)
        valid_epoch_acc /= len(valid_loader)

        # Append metrics to numpy arrays for plotting
        per_epoch_train_acc = np.append(per_epoch_train_acc, round(train_epoch_acc, 5))
        per_epoch_train_loss = np.append(per_epoch_train_loss, round(train_epoch_loss, 5))
        per_epoch_valid_acc = np.append(per_epoch_valid_acc, round(valid_epoch_acc, 5))
        per_epoch_valid_loss = np.append(per_epoch_valid_loss, round(valid_epoch_loss, 5))

        # Display metrics after each Epoch
        print(Fore.GREEN + f"\t[Train] BCE Loss: {train_epoch_loss:.5f}\n")
        print(Fore.GREEN + f"\t[Train] Accuracy: {train_epoch_acc:.5f}\n")
        print(Fore.GREEN + f"\t[Validation] BCE Loss: {valid_epoch_loss:.5f}\n")
        print(Fore.GREEN + f"\t[Validation] Accuracy: {valid_epoch_acc:.5f}\n")

        # Calculate Epoch Time
        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)
        print(Fore.YELLOW + f"\tElapsed Time: {mins} mins, {secs} secs\n")
        current_epoch = epoch + 1


        # MODELCHECKPOINT
        if valid_epoch_loss < best_valid_loss:
            print(Fore.CYAN + f"[MODELCHECKPOINT]: Validation Loss Improved from {best_valid_loss:.5f} to {valid_epoch_loss:.5f}. Saving CheckPoint: {model_name}\n")
            best_valid_loss = valid_epoch_loss
            torch.save(model.state_dict(), model_name)
            counter = 0
        else:
            counter += 1
            if use_lr_schedule and es_patience > 0 and counter > es_patience:
                next_rate = next(learning_rate)
                if next_rate != 0:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = next_rate
                    print(Fore.MAGENTA + f"[LR SCHEDULE]: Learning Rate Reduced to: {next_rate}\n")
                    counter = 0
        # EARLYSTOPPING
        if counter > es_patience and es_patience > 0:
            print(Fore.RED + f"[EARLYSTOPPING]: Validation Loss did not improve from {best_valid_loss:.5f}\n")
            break
    
    # Plot Metric Reslts after training
    if produce_plots:
        plot_results(current_epoch, per_epoch_train_acc, "[Train] Accuracy")
        plot_results(current_epoch, per_epoch_train_loss, "[Train] BCE Loss")
        plot_results(current_epoch, per_epoch_valid_acc, "[Validation] Accuracy")
        plot_results(current_epoch, per_epoch_valid_loss, "[Validation] BCE Loss")

    print(Fore.GREEN + "------------------------ Training Finished ------------------------")




def test(model, test_loader, device, model_name):
    colorama.init(autoreset=True)

    THRESHOLD = 0.5

    test_loader.dataset.LoadInMemory()
    print("\n")

    test_total_loss = 0.0
    test_total_acc = 0.0

    jaccard = 0.0
    f1 = 0.0
    recall = 0.0
    precision = 0.0
    pxl_acc = 0.0
    cohen_kappa = 0

    bce_loss = nn.BCELoss()
    accuracy = BinaryAccuracy()
    ck = BinaryCohenKappa()

    index = 1

    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            print(Fore.YELLOW + f"Predicting: {index} / {len(test_loader)} Steps", end='\r')
            test_img = test_data[0].to(device, dtype=torch.float32)
            test_label = test_data[1].to(device, dtype=torch.float32)

            test_prediction = model(test_img)
            test_loss = bce_loss(test_prediction, test_label)
            test_total_loss += test_loss.item()
            test_acc = accuracy(test_prediction.cpu(), test_label.cpu())
            test_total_acc += test_acc.item()

            score_jaccard, score_f1, score_recall, score_precision = calculate_metrics(test_prediction, test_label, THRESHOLD)

            jaccard += score_jaccard
            f1 += score_f1
            recall += score_recall
            precision += score_precision
            cohen_kappa += ck(test_prediction.cpu(), test_label.cpu())

            index += 1

        test_total_acc /= len(test_loader)
        test_total_loss /= len(test_loader)

        jaccard /= len(test_loader)
        f1 /= len(test_loader)
        recall /= len(test_loader)
        precision /= len(test_loader)
        pxl_acc /= len(test_loader)
        cohen_kappa /= len(test_loader)

    print(f"\n\nModel: {model_name}") 
    print(Fore.CYAN + f"\nAccuracy: {test_total_acc:.5f} | BCE Loss: {test_total_loss:.5f}\n")
    print(Fore.GREEN + f"Jaccard: {jaccard:.5f} | F1: {f1:.5f} | Recall: {recall:.5f} | Precision: {precision:.5f} | Cohen Kappa: {cohen_kappa:.5f}\n")




"""

---------------------------------- MISCELLANOUS ----------------------------------


"""


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    mins = int(elapsed_time / 60)
    secs = int(elapsed_time - (mins * 60))

    return mins, secs


def plot_results(epochs: int, metric_arr: np.array, metric_name: str):

    x = [x for x in range(epochs)]
    x = np.array(x)

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(x, metric_arr, color=(0, 0.8, 1))
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.savefig(f'{metric_name}.png')




def yield_learning_rate(learning_rates: list[float] = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]):
    index = 0
    while True:
        if index < len(learning_rates):
            yield learning_rates[index]
            index += 1
            continue
        yield 0



"""

---------------------------------- ACCURACY & LOSS ----------------------------------


"""



def calculate_metrics(y_pred: torch.tensor, y_true: torch.tensor, threshold: float):
    jaccard = BinaryJaccardIndex(threshold=threshold)
    f1 = BinaryF1Score(threshold=threshold)
    precision = BinaryPrecision(threshold=threshold)
    recall = BinaryRecall(threshold=threshold)

    return jaccard(y_pred.detach().cpu(), y_true.detach().cpu()), f1(y_pred.detach().cpu(), y_true.detach().cpu()), recall(y_pred.detach().cpu(), y_true.detach().cpu()), precision(y_pred.detach().cpu(), y_true.detach().cpu())
