#https://github.com/LightersWang/3DUNet-BraTS-PyTorch/blob/master/configs.py

import numpy as np
import pandas as pd
import torch
import random
import socket
import torch.nn as nn
import time
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import copy
from loss import SoftDiceBCEWithLogitsLoss
from torch import Tensor
from sklearn.preprocessing import PowerTransformer
from MedNextV1 import MedNeXt
import math
import pytorch_warmup as warmup
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import pickle

def dice(output:Tensor, target:Tensor, eps: float=1e-5) -> np.ndarray:
    """calculate multilabel batch dice"""
    target = target.float()
    num = 2 * (output * target).sum(dim=(2,3,4)) + eps
    den = output.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + eps
    dsc = num / den

    return dsc.cpu().numpy()


class CustomImageDataset(Dataset):
    def __init__(self, patient_ids):
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        return idx



def load_data_for_patient(patient_num):
    # Get label and radiomics input
    label = training_labels[patient_num]
    label = torch.tensor(label).long()


    flair_image = np.load((os.path.join(data_dir, str(patient_num), "FLAIR", "preprocessed_FLAIR.npy")))
    flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))
    flair_input = torch.tensor(flair_image).float().unsqueeze(0)


    # Return the result
    result = {
        "flair_input": flair_input,
        "label": label,
    }
    return result


# Seeding
def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    print("SCRIPT IS RUNNING")
    begin = time.time()
    num_trials = 25
    train_proportion = 0.8
    validation_proportion = 0.1
    batch_size_seg = 2
    batch_size_class = 8
    scaling_type = "power" #"standard" or "power"
    dev_mode = False  # if true, use just a small number of patients, to quickly run through and make sure everything is working as expected
    num_important_features = 10

    early_stop = 10
    block_counts = [2,2,2,2,2,2,2,2,2]
    n_channels = 8




    #This is a subset of the best without pretraining found previously
    #Testing to see if this can lower the variance of the difference, resulting in a statistically
    #significant difference, by getting rid of the couple of trials where the model without pretraining
    #did really bad (AUC around 0.5). This should decrease the variance massively, hopefully making the difference
    #statistically significant
    class_param_grid = ParameterGrid([
        {
            # Classification hyperparameters
            "lrs": [0.00001, 0.0001, 0.001],
            "cosine_length": [200,],
            "num_restarts": [1],
            "fc_dropout_rates": [0.75],
            "conv_dropout_rates": [0.25,],
            "noise_factor": [0.1],
            "label_smoothing": [0.0],
            "do_res": [0],
            "local_drop_path_rate": [0.0],
            "warmup_epochs": [3],
            "weight_decay": [1.0]
        },
    ])




    if dev_mode == True:
        num_trials= 2


        class_param_grid = ParameterGrid([
            {
                # Classification hyperparameters
                "lrs": [0.1],
                "cosine_length": [25],
                "num_restarts": [2],
                "fc_dropout_rates": [0.75],
                "conv_dropout_rates": [0.25],
                "noise_factor": [0.1, 0.2],
                "label_smoothing": [0,],
                "do_res": [0],
                "local_drop_path_rate": [0.0],
                "warmup_epochs":[2],
                "weight_decay": [0]
            },
        ])

    ############################################Data Preprocessing######################################################
    # Pointing the excel file which contain the data labels
    if socket.gethostname()=='RT6248W-NMH':
        data_SK_input = pd.read_csv(
            r"Z:/Datasets/MedicalImages/BrainData/SickKids/pLGG_EN_Nov2023/pLGG_4cohorts_532subs.csv")
        if dev_mode:
            data_dir = "C:/Users/kareem kudus/Documents/preprocessed_pLGG_EN_Nov2023"
        else:
            data_dir = "C:/Users/kareem kudus/Documents/preprocessed_pLGG_EN_Nov2023"
    else:
        data_SK_input = pd.read_csv("pLGG_4cohorts_532subs.csv")
        data_dir = "preprocessed_pLGG_EN_Nov2023"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    #######################################Preprocessing the radiomics spreadsheet######################################
    nanmask = np.isnan(data_SK_input["Gen_marker"])
    data_SK = data_SK_input[~nanmask]
    data_SK = data_SK.reindex()
    patients_folders = os.listdir(data_dir)

    def convert_gen_marker(x):
        if x == 2:
            return 2
        elif x ==1:
            return 1
        else:
            return 0

    def convert_location(x):
        if x ==1:
            return 1
        else:
            return 0

    training_labels = {}
    location_labels = {}
    for index, row in data_SK.iterrows():
        if row["folder_name"] in patients_folders:
            training_labels[(row["folder_name"])] = convert_gen_marker(row["Gen_marker"])
            location_labels[(row["folder_name"])] = convert_location(row["Location_1"])
    print(f"Number of patients to use: {len(training_labels)}")

    # Load the dataset into memory
    load_data_time = time.time()
    new_patients_to_use = [] # Keep track of the patients we actually are using
    data_flair_images = None
    data_labels = None
    if dev_mode == True:
        num = 53
    else:
        num = len(training_labels)
    for i in range(num):
        print(f"{list(training_labels.keys())[i]} ({i}/{num})") #The patient id
        result = load_data_for_patient(list(training_labels.keys())[i]) #A dictionary containing data for this patient
        # print(result)
        if result != None:
            if data_labels == None:
                data_labels = torch.unsqueeze(result["label"], 0)
                data_flair_images = torch.unsqueeze(result["flair_input"],0)

            else:
                data_labels = torch.cat((data_labels, torch.unsqueeze(result["label"], 0)))
                data_flair_images = torch.cat((data_flair_images,torch.unsqueeze(result["flair_input"],0)))

            new_patients_to_use.append(list(training_labels.keys())[i])

        else:
            sys.exit(f"Had an issue loading data for patient {list(training_labels.keys())[i]}")
    patients_to_use = new_patients_to_use
    print(f"Time to load data into memory: {time.time() - load_data_time}")
    print(f"Number of patients loaded into memory: {len(patients_to_use)}")

    ###############################################Training#############################################################
    print(f"Total number of segmentation hyperparameter configurations: {len(list(class_param_grid))}")

    # Results file for this set of hyperparameters
    name = None #reset file name, overwriting last name from previous LR

    # Variables to hold results from each trial



    #Classification
    class_validation_AUCs = []
    class_test_AUCs = []
    class_train_AUCs = []
    class_test_accs = []
    class_best_epochs = []
    class_best_lrs = []
    class_best_fc_dropout_rates = []
    class_best_conv_dropout_rates = []
    class_best_noises = []
    class_best_cosine_lengths = []
    class_best_label_smoothings = []
    class_best_test_auc_breakdowns =[]
    class_best_test_confusion_matrixs = []
    class_best_test_label_trues = []
    class_best_test_label_estimateds = []
    class_best_local_drop_path_rates = []
    class_best_do_ress = []
    class_best_warmup_epochs = []
    class_best_decay_rates = []


    trial_time = []

    for t in range(num_trials):

        print(f"Trial #{str(t)}")
        time_being_trial = time.time()
        # Set the seed for this iteration
        if t == 0:
            random_seed(10  , True)
            next_seed = random.randint(0,100000)
        else:
            random_seed(next_seed, True)
            next_seed = random.randint(0, 100000)
        print(next_seed)

        # Prepare the data loaders
        dataset = CustomImageDataset(patients_to_use)


        # Determine number of points in each subset
        train_size = int(train_proportion * len(dataset))
        validation_size = int(validation_proportion * len(dataset))
        test_size = len(dataset) - train_size - validation_size
        print(f"Number of training points: {train_size}")
        print(f"Number of validation points: {validation_size}")
        print(f"Number of test points: {test_size}")

        # Split the dataset
        indices = []
        patient_labels = []
        patient_ids = []
        for i in range(len(dataset)):
            patient_labels.append(data_labels[i].item())
            patient_ids.append(dataset.patient_ids[i])
            indices.append(i)
        train_indices, test_and_val_indices = train_test_split(indices, train_size=train_size, random_state=next_seed,
                                                               stratify=patient_labels)

        val_indices, test_indices = train_test_split(test_and_val_indices, train_size=validation_size,
                                                     random_state=next_seed,
                                                     stratify=[patient_labels[i] for i in test_and_val_indices])
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        validation_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        if (bool(set(train_dataset) & set(validation_dataset)) or bool(set(train_dataset) & set(test_dataset)) or bool(
                set(validation_dataset) & set(test_dataset))):
            exit("There is overlap between our datasets")

        temp_labels = data_labels[train_indices]
        print(f"Number of training points: {temp_labels.shape[0]}, ")
        temp_labels = data_labels[val_indices]
        print(f"Number of validation points: {temp_labels.shape[0]}, ")
        temp_labels = data_labels[test_indices]
        print(f"Number of test points: {temp_labels.shape[0]}, ")


        # Determine class distribution among the different datasets
        temp_labels = data_labels[train_indices]
        print(f"Number of training points: {temp_labels.shape[0]}, "
              f"proportion of class 0/1/2: "
              f"{round(temp_labels[temp_labels==0].size(dim=0) / temp_labels.size(dim=0), 3)}/"
              f"{round(temp_labels[temp_labels==1].size(dim=0) / temp_labels.size(dim=0), 3)}/"
              f"{round(temp_labels[temp_labels==2].size(dim=0) / temp_labels.size(dim=0), 3)}")
        temp_labels = data_labels[val_indices]
        print(f"Number of validation points: {temp_labels.shape[0]}, "
              f"proportion of class 0/1/2: "
              f"{round(temp_labels[temp_labels == 0].size(dim=0) / temp_labels.size(dim=0), 3)}/"
              f"{round(temp_labels[temp_labels == 1].size(dim=0) / temp_labels.size(dim=0), 3)}/"
              f"{round(temp_labels[temp_labels == 2].size(dim=0) / temp_labels.size(dim=0), 3)}")

        temp_labels = data_labels[test_indices]
        print(f"Number of test points: {temp_labels.shape[0]}, "
              f"proportion of class 0/1/2: "
              f"{round(temp_labels[temp_labels==0].size(dim=0) / temp_labels.size(dim=0), 3)}/"
              f"{round(temp_labels[temp_labels==1].size(dim=0) / temp_labels.size(dim=0), 3)}/"
              f"{round(temp_labels[temp_labels==2].size(dim=0) / temp_labels.size(dim=0), 3)}")



        ##########################################Train the Classification Model##########################################
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_class, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size_class, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_class, shuffle=True)

        class_lowest_val_loss_overall = 10000000000.0 # Lowest loss on validation set over all of they hyper configs
        class_lowest_val_loss_epoch = 0  # Epoch on which lowest loss on validation set was acheived
        class_lowest_val_loss_hyper_config = None
        class_best_train_auc = None
        class_best_val_auc = None  # The AUC on the validation set for the model that had the best validation loss
        class_best_test_auc = None  # The AUC on the validation set for the model that had the best validation loss
        class_best_test_acc = None
        class_best_model = None
        class_best_lr = None
        class_best_fc_dropout_rate = None
        class_best_conv_dropout_rate = None
        class_best_noise = None
        class_best_cosine_length = None
        class_best_label_smoothing = None
        class_best_do_res = None
        class_best_local_drop_path_rate = None
        class_best_decay_rate = None
        class_best_warmup_epoch = None


        class_best_test_auc_breakdown = None
        class_best_test_confusion_matrix = None
        class_best_test_label_true = None
        class_best_test_label_estimated = None

        for hyper_counter in range(len(list(class_param_grid))):
            print("\n")
            print(f"Hyperparameter config {hyper_counter + 1}/{len(list(class_param_grid))}")
            hypers = class_param_grid[hyper_counter]
            print(hypers)

            if early_stop:
                lowest_val_loss_this_hyper = 10000000000.0
                lowest_val_loss_epoch_this_hyper = 0

            # Hyperparameters for training CNN
            class_lr = hypers["lrs"]
            class_cosine_length = hypers["cosine_length"]
            class_num_restarts = hypers["num_restarts"]
            class_noise_factor = hypers["noise_factor"]
            class_fc_dropout_rate = hypers["fc_dropout_rates"]
            class_conv_dropout_rate = hypers["conv_dropout_rates"]
            class_label_smoothing = hypers["label_smoothing"]
            class_do_res = hypers["do_res"]
            class_local_drop_path_rate = hypers["local_drop_path_rate"]
            class_weight_decay = hypers["weight_decay"]
            class_warmup_epochs = hypers["warmup_epochs"]

            # Need to calculate the value of certain params for this trial
            class_epochs = class_cosine_length * class_num_restarts
            if dev_mode == True:
                class_epochs = 2

            # Define neural network stuff
            track_time_start = time.time()
            net = MedNeXt(
                in_channels=1,
                n_channels=n_channels,
                n_classes=1,
                exp_r=2,
                kernel_size=3,
                deep_supervision=False,
                do_res=class_do_res,
                do_res_up_down=False,
                block_counts=block_counts,
                seg=False,
                reg=False,
                fc_dropout_rate=class_fc_dropout_rate,
                conv_dropout_rate=class_conv_dropout_rate,
                local_drop_path_rate=class_local_drop_path_rate,
                reg_num_outputs = num_important_features
            )
            pretrain = torch.load("pretrained_model0.pt", map_location='cpu')
            net_dict = net.state_dict()
            pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys() and "fc" not in k}
            net_dict.update(pretrain_dict)
            net.load_state_dict(net_dict)
            net.to(device)


            temp_labels=None
            for batch_idx, (idx) in enumerate(train_dataloader):
                label = data_labels[idx].to(device)
                if temp_labels == None:
                    temp_labels = label
                else:
                    temp_labels = torch.cat([temp_labels, label])
            num_patients_total = temp_labels.shape[0]
            loss_weights = torch.tensor([
                num_patients_total/temp_labels[temp_labels == 0].size(dim=0),
                num_patients_total/temp_labels[temp_labels == 1].size(dim=0),
                num_patients_total/temp_labels[temp_labels == 2].size(dim=0)]).to(device)
            loss_weights = loss_weights/torch.sum(loss_weights)
            criterion = nn.CrossEntropyLoss(weight = loss_weights, label_smoothing=class_label_smoothing)
            optimizer = optim.AdamW(net.parameters(), lr=class_lr, weight_decay=class_weight_decay)
            softmax = torch.nn.Softmax(dim=0)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=class_cosine_length, T_mult=1, eta_min=class_lr/100 )
            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=class_warmup_epochs)

            optimizer.zero_grad()
            for epoch in range(class_epochs):
                temp_outer = time.time()
                net.train()
                train_loss = 0
                training_true = []
                training_estimated = []
                iters = len(train_dataloader)

                for batch_idx, (idx) in enumerate(train_dataloader):

                    # Put data on GPU
                    inputs = data_flair_images[idx].to(device)
                    label = data_labels[idx].to(device)

                    # Add noise to images
                    noise = torch.randn_like(inputs, device=device) * class_noise_factor
                    inputs = inputs + noise
                    # Forward + Backward + Optimize
                    output = net(inputs)
                    loss = criterion(output, label)
                    loss.backward()


                    if ((batch_idx + 1) % 1 == 0) or (batch_idx + 1 == len(train_dataloader)):
                        optimizer.step()
                        optimizer.zero_grad()

                    for i in range(len(label.tolist())):
                        training_true.append(label[i].cpu().detach().numpy().item())
                        training_estimated.append(softmax(output[i]).cpu().detach().numpy())

                    # Keep track of loss through the entire epoch
                    train_loss += loss.item() * inputs.shape[0]

                # Calculate average loss over epoch
                train_loss = train_loss / len(training_true)

                # Get results-flair on the validation and test sets
                net.eval()
                with torch.set_grad_enabled(False):
                    # Validation
                    validation_true = []
                    validation_estimated = []
                    val_loss = 0
                    # Put data on GPU
                    for batch_idx, (idx) in enumerate(validation_dataloader):
                        inputs, label = data_flair_images[idx].to(device), data_labels[idx].to(device)
                        output = net(inputs)
                        for i in range(len(label.tolist())):
                            validation_true.append(label[i].cpu().detach().numpy().item())
                            validation_estimated.append(softmax(output[i]).cpu().detach().numpy())
                        loss = criterion(output, label)
                        val_loss += loss.item()*inputs.shape[0]
                    val_loss = val_loss / len(validation_true)


                    # Test
                    test_true = []
                    test_estimated = []
                    test_estimated_label = []
                    # Put data on GPU
                    for batch_idx, (idx) in enumerate(test_dataloader):
                        inputs, label = data_flair_images[idx].to(device), data_labels[idx].to(device)
                        output = net(inputs)
                        for i in range(len(label.tolist())):
                            test_true.append(label[i].cpu().detach().numpy().item())
                            test_estimated.append(softmax(output[i]).cpu().detach().numpy().tolist())
                            test_estimated_label.append(torch.argmax(softmax(output[i])).cpu().detach().numpy())


                # Calculate the AUC for the different models
                # print(training_true)
                # print(training_estimated)
                # print(len(training_true))
                # print(len(training_estimated))
                train_auc = roc_auc_score(training_true, training_estimated, multi_class="ovr", average="macro")
                val_auc = roc_auc_score(validation_true, validation_estimated, multi_class="ovr", average="macro")
                test_auc = roc_auc_score(test_true, test_estimated, multi_class="ovr", average="macro")
                test_acc = accuracy_score(test_true, test_estimated_label)
                test_auc_breakdown = roc_auc_score(test_true, test_estimated, multi_class="ovr", average=None)
                test_confusion_matrix = confusion_matrix(test_true, test_estimated_label).tolist()

                if val_loss< class_lowest_val_loss_overall:
                    class_lowest_val_loss_overall = val_loss
                    class_lowest_val_loss_epoch = epoch
                    class_best_test_acc = test_acc
                    class_best_test_auc = test_auc
                    class_best_val_auc = val_auc
                    class_best_test_auc_breakdown = test_auc_breakdown
                    class_best_train_auc = train_auc
                    class_best_model = copy.deepcopy(net)
                    class_lowest_val_loss_hyper_config = hyper_counter
                    class_best_lr = hypers["lrs"]
                    class_best_fc_dropout_rate = hypers["fc_dropout_rates"]
                    class_best_conv_dropout_rate = hypers["conv_dropout_rates"]
                    class_best_noise = class_noise_factor
                    class_best_cosine_length = class_cosine_length
                    class_best_label_smoothing = hypers["label_smoothing"]
                    class_best_test_confusion_matrix = test_confusion_matrix
                    class_best_test_label_true = test_true
                    class_best_test_label_estimated = test_estimated
                    class_best_local_drop_path_rate = class_local_drop_path_rate
                    class_best_do_res = class_do_res
                    class_best_warmup_epoch = class_warmup_epochs
                    class_best_decay_rate = class_weight_decay

                if early_stop:
                    if val_loss < lowest_val_loss_this_hyper:
                        lowest_val_loss_this_hyper = val_loss
                        lowest_val_loss_epoch_this_hyper = epoch

                epoch_result_string = f"trial: {t}, epoch: {epoch}, training loss {round(train_loss, 3)} , " \
                                      f"validation loss: {round(val_loss, 3)}, " \
                                      f"training AUC: {round(train_auc, 3)}, validation AUC: {round(val_auc, 3)}, " \
                                      f"test AUC: {round(test_auc, 3)}, learning rate: {round(optimizer.param_groups[0]['lr'], 5)}, " \
                                      f"Lowest val loss: {round(class_lowest_val_loss_overall, 3)}, on epoch {class_lowest_val_loss_epoch}," \
                                      f"of hyperparam config {class_lowest_val_loss_hyper_config}"

                if early_stop:
                    epoch_result_string += f", lowest val loss this hyper: {round(lowest_val_loss_this_hyper, 3)}, on epoch {lowest_val_loss_epoch_this_hyper}"
                print(epoch_result_string)

                if early_stop:
                    if epoch - lowest_val_loss_epoch_this_hyper > early_stop:
                        print("Stopped early!")
                        break
                with warmup_scheduler.dampening():
                    scheduler.step()

        torch.cuda.synchronize()
        print("Finished training Classifier")
        print(f"Time Classifier: {round(time.time() - track_time_start, 3)}")

        class_validation_AUCs.append(round(class_best_val_auc, 3))
        class_test_AUCs.append(round(class_best_test_auc, 3))
        class_train_AUCs.append(round(class_best_train_auc, 3))
        class_test_accs.append(round(class_best_test_acc,3))
        class_best_epochs.append(round(class_lowest_val_loss_epoch, 3))
        class_best_lrs.append(class_best_lr)
        class_best_fc_dropout_rates.append(class_best_fc_dropout_rate)
        class_best_conv_dropout_rates.append(class_best_conv_dropout_rate)
        class_best_cosine_lengths.append(class_best_cosine_length)
        class_best_label_smoothings.append(class_best_label_smoothing)
        class_best_noises.append(class_best_noise)
        class_best_test_auc_breakdowns.append(class_best_test_auc_breakdown)
        class_best_test_confusion_matrixs.append(class_best_test_confusion_matrix)
        class_best_test_label_trues.append(class_best_test_label_true)
        class_best_test_label_estimateds.append(class_best_test_label_estimated)
        class_best_do_ress.append(class_best_do_res)
        class_best_local_drop_path_rates.append(class_best_local_drop_path_rate)
        class_best_decay_rates.append(class_best_decay_rate)
        class_best_warmup_epochs.append(class_best_warmup_epoch)

        del net, label, output, val_loss, inputs, train_loss, loss, optimizer, noise
        torch.cuda.empty_cache()

        print(f"Time for this trial: {round(time.time() - time_being_trial,3)}")

        trial_time.append(round(time.time() - time_being_trial,3))

        # Save dictionary of predictions
        timestamp = time.strftime("%m%d-%H%M")

        cols = []
        cols = cols + ["Class Train AUCs",
                    "Class Validation AUCs",
                    "Class Test AUCs",
                    "Class Test AUC breakdowns",
                    "Class Test Accuracy",
                    "Class Test Confusion Matrix",
                    "Class Test True Labels",
                    "Class Test Estimated Labels",
                    "Class best epoch",
                    "Class learning_rate",
                    "Class fc_dropout_rate",
                    "Class conv_dropout_rate",
                    "Class noise",
                    "Class cosine length",
                    "Class label smoothing",
                    "Class local drop path rate",
                    "Class do res",
                    "Class weight decay",
                    "Class warmup epochs",
                    "Times"]
        results = pd.DataFrame(
            columns=[cols])

        print(results)
        for i in range(len(class_validation_AUCs)):
            print(i)
            result = []
            result.append(class_train_AUCs[i])
            result.append(class_validation_AUCs[i])
            result.append(class_test_AUCs[i])
            result.append(class_best_test_auc_breakdowns[i])
            result.append(class_test_accs[i])
            result.append(class_best_test_confusion_matrixs[i])
            result.append(class_best_test_label_trues[i])
            result.append(class_best_test_label_estimateds[i])
            result.append(class_best_epochs[i])
            result.append(class_best_lrs[i])
            result.append(class_best_fc_dropout_rates[i])
            result.append(class_best_conv_dropout_rates[i])
            result.append(class_best_noises[i])
            result.append(class_best_cosine_lengths[i])
            result.append(class_best_label_smoothings[i])
            result.append(class_best_local_drop_path_rates[i])
            result.append(class_best_do_ress[i])
            result.append(class_best_decay_rates[i])
            result.append(class_best_warmup_epochs[i])
            result.append(trial_time[i])
            print(result)
            results.loc[len(results)] = result
            print(results)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(results)

        try:
            os.remove(name)
            print(f"Removed file {name}")
        except:
            print("No results file to remove")

        try:
            os.remove("preds_" + name)
            print(f"Removed file {'preds_' + name}")
        except:
            print("No predictions file to remove")



        name = ""


        name = name + "_classauc" + str(np.round(np.mean(class_test_AUCs), 3)) + "_classacc" +str(np.round(np.mean(class_test_accs),3)) +\
               "_npts" +str(len(dataset))  + \
               "_" + timestamp + ".csv"


        name = "plgg_whole_image_"\
               + "_results" + "_t" + str(t+1) + name
        print(name)
        results.to_csv(name)







