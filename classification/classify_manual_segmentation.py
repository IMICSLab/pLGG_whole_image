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

    location = location_labels[patient_num]
    location = torch.tensor(location).float()

    # #Radiomics
    radiomics = df_rad.loc[patient_num].values
    radiomics = torch.FloatTensor(radiomics)

    # Load the segmentation data
    mask = np.load(os.path.join(data_dir, str(patient_num), "FLAIR", "preprocessed_segmentation.npy"))


    flair_image = np.load((os.path.join(data_dir, str(patient_num), "FLAIR", "preprocessed_FLAIR.npy")))
    flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))
    flair_input = torch.tensor(np.multiply(flair_image, mask)).float().unsqueeze(0)
    mask = torch.tensor(mask).float().unsqueeze(0)

    # Return the result
    result = {
        "flair_input": flair_input,
        "label": label,
        "mask":mask,
        "radiomics":radiomics,
        "location":location
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
    num_trials = 30
    train_proportion = 0.8
    validation_proportion = 0.1
    batch_size_seg = 2
    batch_size_class = 8
    pretrain_seg = False
    pretrain_rad = False
    scaling_type = "power" #"standard" or "power"
    dev_mode = False  # if true, use just a small number of patients, to quickly run through and make sure everything is working as expected
    num_important_features = 10
    # early_stop = 20
    early_stop = 10
    save_predictions = False
    save_segmentations = False
    bayesian_inference = False
    block_counts = [2,2,2,2,2,2,2,2,2]
    n_channels = 8

    # radiomics_name_list = [
    #             "original_shape_SurfaceVolumeRatio",
    #             "original_gldm_DependenceNonUniformityNormalized",
    #             "original_shape_Flatness",
    #             "original_shape_Sphericity",
    #             "original_glszm_ZonePercentage",
    #         ]

    seg_param_grid = ParameterGrid([
        #This did decently, but there was crazy overfitting for radiomics
        {
            # Segmentation hyperparameters
            "lrs": [0.01],
            "cosine_length": [50],
            "num_restarts": [1],
            "fc_dropout_rates": [0.75],
            "conv_dropout_rates": [0.25],
            "noise_factor": [0.1,],
            "global_drop_path_rate": [0],
            "do_res": [0],
            "local_drop_path_rate": [0],
            "warmup_epochs":[3],
            "weight_decay": [0.1]
        },
    ])

    # reg_param_grid = ParameterGrid([
    #     {
    #         # Regression hyperparameters
    #         "lrs": [0.001,],
    #         "cosine_length": [50,],
    #         "num_restarts": [1,],
    #         "fc_dropout_rates": [0.75],
    #         "conv_dropout_rates": [0.25],
    #         "noise_factor": [0.1],
    #         "do_res": [0],
    #         "local_drop_path_rate": [0.0],
    #         "warmup_epochs": [3],
    #         "weight_decay": [0.1]
    #     },
    # ])

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

    #This was the best so far with pretraining
    # class_param_grid = ParameterGrid([
    #     {
    #         # Classification hyperparameters
    #         "lrs": [0.0001, 0.00001],
    #         "cosine_length": [200,],
    #         "num_restarts": [1],
    #         "fc_dropout_rates": [0.5,0.75],
    #         "conv_dropout_rates": [0.25,],
    #         "noise_factor": [0.1],
    #         "label_smoothing": [0.0],
    #         "do_res": [0],
    #         "local_drop_path_rate": [0.0],
    #         "warmup_epochs": [3],
    #         "weight_decay": [1.0, 0.001]
    #     },
    # ])

    # THESE ARE FOR WITHOUT PRETRAINING
    # class_param_grid = ParameterGrid([
    #     {
    #         # Classification hyperparameters
    #         "lrs": [0.0001],
    #         "cosine_length": [200,],
    #         "num_restarts": [1],
    #         "fc_dropout_rates": [0.5],
    #         "conv_dropout_rates": [0.25],
    #         "noise_factor": [0.1],
    #         "label_smoothing": [0.0],
    #         "do_res": [0],
    #         "local_drop_path_rate": [0.0],
    #         "warmup_epochs": [3],
    #         "weight_decay": [0.1]
    #     },
    #     {
    #         # Classification hyperparameters
    #         "lrs": [0.001],
    #         "cosine_length": [50],
    #         "num_restarts": [1],
    #         "fc_dropout_rates": [0.5],
    #         "conv_dropout_rates": [0.25],
    #         "noise_factor": [0.1],
    #         "label_smoothing": [0.0],
    #         "do_res": [0],
    #         "local_drop_path_rate": [0.0],
    #         "warmup_epochs": [3],
    #         "weight_decay": [0.001, 0.1]
    #     },
    # ])


    if dev_mode == True:
        num_trials= 10
        seg_param_grid = ParameterGrid([
            {
                # Segmentation hyperparameters
                "lrs": [0.001],
                "cosine_length": [50],
                "num_restarts": [1],
                "fc_dropout_rates": [0.75],
                "conv_dropout_rates": [0.25],
                "noise_factor": [0.1],
                "global_drop_path_rate": [0],
                "do_res": [0],
                "local_drop_path_rate": [0],
                "warmup_epochs": [3],
                "weight_decay": [0.1]
            },
        ])

        reg_param_grid = ParameterGrid([
            {
                # Regression hyperparameters
                "lrs": [0.1],
                "cosine_length": [25],
                "num_restarts": [2],
                "fc_dropout_rates": [0.25],
                "conv_dropout_rates": [0.25],
                "noise_factor": [0.1],
                "do_res": [1],
                "local_drop_path_rate": [0.5],
                "warmup_epochs":[2],
                "weight_decay": [0.01]
            },
        ])

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
    print(data_SK_input.shape)
    data_SK = data_SK_input[~nanmask]
    print(data_SK.shape)
    print(data_SK_input[nanmask])
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

    #Radiomics features
    df_rad = pd.read_csv("Radiomics_binWidth-25_NoNormalization_Whole-Tumor_flair.csv")
    print(df_rad.shape)
    df_rad = df_rad.drop(df_rad.columns[[0]+[x for x in range(2,29)]], axis =1)
    df_rad.set_index('Patient_ID', inplace=True)
    df_rad = df_rad.filter(like='original', axis=1)
    # dict_rad_features = {}
    print(df_rad.shape)

    # for i in range(len(list(df_rad["Patient_ID"].values))):
    #     temp = []
    #     for l in radiomics_name_list:
    #         temp.append(list(df_rad[l].values)[i])
    #     temp.append(data_SK[data_SK["code"]==list(df_rad["Patient_ID"].values)[i]]["location"].values[0])
    #     dict_rad_features[list(df_rad["Patient_ID"].values)[i]] = temp
    # # print(dict_rad_features)

    # Load the dataset into memory
    load_data_time = time.time()
    new_patients_to_use = [] # Keep track of the patients we actually are using
    data_flair_images = None
    data_masks = None
    data_labels = None
    data_radiomics = None
    data_location = None
    if dev_mode == True:
        num = 53
    else:
        num = len(training_labels)
    for i in range(num):
        print(f"{list(training_labels.keys())[i]} ({i}/{num})") #The patient id
        result = load_data_for_patient(list(training_labels.keys())[i]) #A dictionary containing data for this patient
        # print(result)
        if result != None:
            if data_masks == None:
                data_labels = torch.unsqueeze(result["label"], 0)
                data_masks = torch.unsqueeze(result["mask"], 0)
                data_flair_images = torch.unsqueeze(result["flair_input"],0)
                data_radiomics = torch.unsqueeze(result["radiomics"], 0)
                data_location = torch.unsqueeze(result["location"], 0)
            else:
                data_labels = torch.cat((data_labels, torch.unsqueeze(result["label"], 0)))
                data_masks = torch.cat((data_masks, torch.unsqueeze(result["mask"], 0)))
                data_flair_images = torch.cat((data_flair_images,torch.unsqueeze(result["flair_input"],0)))
                data_radiomics = torch.cat((data_radiomics, torch.unsqueeze(result["radiomics"], 0)))
                data_location = torch.cat((data_location, torch.unsqueeze(result["location"], 0)))
            new_patients_to_use.append(list(training_labels.keys())[i])

        else:
            sys.exit(f"Had an issue loading data for patient {list(training_labels.keys())[i]}")
    patients_to_use = new_patients_to_use
    print(f"Time to load data into memory: {time.time() - load_data_time}")
    print(f"Number of patients loaded into memory: {len(patients_to_use)}")

    ###############################################Training#############################################################
    print(f"Total number of segmentation hyperparameter configurations: {len(list(seg_param_grid))}")

    # Results file for this set of hyperparameters
    name = None #reset file name, overwriting last name from previous LR

    # Variables to hold results from each trial

    # CNN
    seg_validation_dices = []
    seg_test_dices = []
    seg_test_aucs = []
    seg_train_dices = []
    seg_best_epochs = []
    seg_best_lrs = []
    seg_best_fc_dropout_rates = []
    seg_best_conv_dropout_rates = []
    seg_best_noises = []
    seg_best_cosine_lengths = []
    seg_best_global_drop_path_rates = []
    seg_best_local_drop_path_rates = []
    seg_best_do_ress = []
    seg_best_warmup_epochs = []
    seg_best_decay_rates = []
    seg_best_radiomics_preds =[]
    seg_best_radiomics_labels =[]
    seg_best_location_preds = []
    seg_best_location_labels = []
    # seg_best_segmentation_preds = []
    # seg_best_segmentation_labels = []

    #Regression
    reg_validation_loss = []
    reg_test_loss = []
    reg_train_loss = []
    reg_best_epochs = []
    reg_best_lrs = []
    reg_best_fc_dropout_rates = []
    reg_best_conv_dropout_rates = []
    reg_best_noises = []
    reg_best_cosine_lengths = []
    reg_best_local_drop_path_rates = []
    reg_best_do_ress = []
    reg_best_warmup_epochs = []
    reg_best_decay_rates = []

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
    bayesian_preds_all = []
    bayesian_labels_all = []

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

        temp_labels = data_masks[train_indices]
        print(f"Number of training points: {temp_labels.shape[0]}, ")
        temp_labels = data_masks[val_indices]
        print(f"Number of validation points: {temp_labels.shape[0]}, ")
        temp_labels = data_masks[test_indices]
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


        ##########################################Train the Segmentation Model##########################################
        if pretrain_seg:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size_seg, shuffle=True)
            validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size_seg, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size_seg, shuffle=False)

            # #Fit Ridge Regression to find most predictive radiomic features
            # X_train = pd.DataFrame(data_radiomics[train_indices].numpy())
            # Y_train = pd.DataFrame(data_labels[train_indices].numpy())
            # X_val = pd.DataFrame(data_radiomics[val_indices].numpy())
            # Y_val = pd.DataFrame(data_labels[val_indices].numpy())
            # X_test = pd.DataFrame(data_radiomics[test_indices].numpy())
            # Y_test = pd.DataFrame(data_labels[test_indices].numpy())
            # rfc_cv = RandomForestClassifier(n_jobs=-1)
            # kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=next_seed)
            # param_grid = {
            #     'n_estimators': [75, 150, 300],
            #     'random_state': [100],
            #     'criterion': ["entropy"],
            #     'min_samples_leaf': [1, 2],
            #     'min_samples_split': [2, 4],
            #     'max_depth': [8],
            #     'max_samples': [0.5, 0.75, None],
            #     'max_features': [0.25, 0.5, 0.75],
            # }
            # CV_rfc = GridSearchCV(estimator=rfc_cv, param_grid=param_grid, cv=kf, verbose=0, scoring="roc_auc_ovr",
            #                       n_jobs=-1,
            #                       return_train_score=True)
            # temp = CV_rfc.fit(X_train, Y_train.squeeze())
            # best_model_rfc = temp.best_estimator_
            # print(round(roc_auc_score(Y_test, best_model_rfc.predict_proba(X_test), multi_class="ovr", average="macro"), 3))
            # perm_return = permutation_importance(best_model_rfc, X_val, Y_val.squeeze(),
            #                                      scoring="neg_log_loss", n_repeats=30, n_jobs=-1,
            #                                      random_state=next_seed)
            # feature_importance = list(perm_return.importances_mean)
            # important_feature_indices = [feature_importance.index(x) for x in
            #                             sorted(feature_importance, reverse=True)[:num_important_features
            #                             ]]
            # print(df_rad.columns[X_train.columns[important_feature_indices]])
            # data_radiomics_temp = pd.DataFrame(data_radiomics.numpy())
            # data_radiomics_temp = data_radiomics_temp.iloc[:,important_feature_indices]
            # data_radiomics_temp = torch.Tensor(data_radiomics_temp.values)
            # if scaling_type =="power":
            #     # Power Transform
            #     means = data_radiomics_temp[train_indices].mean(0)
            #     stdevs = data_radiomics_temp[train_indices].std(0)
            #     data_radiomics_temp = (data_radiomics_temp - means) / stdevs
            #     scaler = PowerTransformer()
            #     scaler.fit(data_radiomics_temp[train_indices].numpy())
            #     data_radiomics_temp = torch.Tensor(scaler.transform(data_radiomics_temp.numpy()))
            # elif scaling_type =="standard":
            #     # Standard Scaler
            #     means = data_radiomics_temp[train_indices].mean(0)
            #     stdevs = data_radiomics_temp[train_indices].std(0)
            #     data_radiomics_temp = (data_radiomics_temp - means) / stdevs
            # else:
            #     sys.exit()


            seg_lowest_val_loss_overall = 10000000000.0 # Lowest loss on validation set over all of they hyper configs
            seg_lowest_val_loss_epoch = 0  # Epoch on which lowest loss on validation set was acheived
            seg_lowest_val_loss_hyper_config = None
            seg_best_train_dice = None
            seg_best_val_dice = None  # The AUC on the validation set for the model that had the best validation loss
            seg_best_test_dice = None  # The AUC on the validation set for the model that had the best validation loss
            seg_best_test_auc = None
            seg_best_model = None
            seg_best_lr = None
            seg_best_conv_dropout_rate = None
            seg_best_fc_dropout_rate = None
            seg_best_noise = None
            seg_best_cosine_length = None
            seg_best_global_drop_path_rate = None
            seg_best_local_drop_path_rate = None
            seg_best_do_res = None
            seg_best_decay_rate = None
            seg_best_warmup_epoch = None
            seg_best_radiomics_pred =None
            seg_best_radiomics_label =None
            seg_best_location_pred = None
            seg_best_location_label = None
            # seg_best_segmentation_pred = None
            # seg_best_segmentation_label = None

            for hyper_counter in range(len(list(seg_param_grid))):
                print("\n")
                print(f"Hyperparameter config {hyper_counter + 1}/{len(list(seg_param_grid))}")
                hypers = seg_param_grid[hyper_counter]
                print(hypers)

                if early_stop:
                    lowest_val_loss_this_hyper = 10000000000.0
                    lowest_val_loss_epoch_this_hyper = 0


                # Hyperparameters for training CNN
                seg_lr = hypers["lrs"]
                seg_cosine_length = hypers["cosine_length"]
                seg_num_restarts = hypers["num_restarts"]
                seg_noise_factor = hypers["noise_factor"]
                seg_fc_dropout_rate = hypers["fc_dropout_rates"]
                seg_conv_dropout_rate = hypers["conv_dropout_rates"]
                seg_global_drop_path_rate = hypers["global_drop_path_rate"]
                seg_local_drop_path_rate = hypers["local_drop_path_rate"]
                seg_do_res = hypers["do_res"]
                seg_warmup_epoch = hypers["warmup_epochs"]
                seg_weight_decay = hypers["weight_decay"]

                # Need to calculate the value of certain params for this trial
                seg_epochs = seg_cosine_length * seg_num_restarts
                if dev_mode == True:
                    seg_epochs = 4

                # Define neural network stuff
                track_time_start = time.time()

                net = MedNeXt(
                    in_channels = 1,
                    n_channels = n_channels,
                    n_classes = 1,
                    exp_r=2,
                    kernel_size=3,
                    deep_supervision=False,
                    do_res=seg_do_res,
                    do_res_up_down = False,
                    block_counts = block_counts,
                    fc_dropout_rate=seg_fc_dropout_rate,
                    conv_dropout_rate = seg_conv_dropout_rate,
                    global_drop_path_rate = seg_global_drop_path_rate,
                    local_drop_path_rate=seg_local_drop_path_rate,
                    reg_num_outputs = num_important_features
                )

                net.to(device)

                criterion_seg = SoftDiceBCEWithLogitsLoss().cuda()
                criterion_loc = torch.nn.BCELoss()
                criterion_rad = nn.SmoothL1Loss(beta = 0.5)
                optimizer = optim.AdamW(net.parameters(), lr=seg_lr, weight_decay=seg_weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=seg_cosine_length, T_mult=1, eta_min=seg_lr/100)
                warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=seg_warmup_epoch)
                softmax = torch.nn.Softmax(dim=0)

                optimizer.zero_grad()
                for epoch in range(seg_epochs):
                    temp_outer = time.time()
                    net.train()
                    train_loss = 0
                    train_bce_loss = 0
                    train_dsc_loss = 0
                    train_loc_loss = 0
                    train_rad_loss = 0
                    train_dice = 0
                    training_true = []
                    training_estimated = []
                    iters = len(train_dataloader)

                    for batch_idx, (idx) in enumerate(train_dataloader):

                        # Put data on GPU
                        inputs = data_flair_images[idx].to(device)
                        label_mask = data_masks[idx].to(device)
                        label_location = data_location[idx].to(device)
                        # label_radiomics = data_radiomics_temp[idx].to(device)

                        # Add noise to images
                        noise = torch.randn_like(inputs, device=device)*seg_noise_factor
                        inputs = inputs+noise
                        # Forward + Backward + Optimize
                        output_seg, output_loc, output_rad = net(inputs)
                        bce_loss, dsc_loss = criterion_seg(output_seg, label_mask)
                        loc_loss = criterion_loc(torch.sigmoid(output_loc), label_location)
                        # rad_loss = criterion_rad(output_rad, label_radiomics)
                        # rad_loss = nn.functional.huber_loss(output_rad, label_radiomics, delta=0.5)
                        loss = bce_loss + dsc_loss + loc_loss #+ #rad_loss
                        # loss = loc_loss
                        loss.backward()
                        if ((batch_idx + 1) % 2 == 0) or (batch_idx + 1 == len(train_dataloader)):
                            optimizer.step()
                            optimizer.zero_grad()


                        # seg_map = torch.where(softmax(output) > 0.5, True, False).detach()
                        # Keep track of loss through the entire epoch
                        train_loss += loss.item()*inputs.shape[0]
                        train_bce_loss += bce_loss.item()*inputs.shape[0]
                        train_dsc_loss += dsc_loss.item() * inputs.shape[0]
                        train_loc_loss += loc_loss.item()* inputs.shape[0]
                        # train_rad_loss += rad_loss.item()*inputs.shape[0]
                        train_dice += (1-dsc_loss.item())*inputs.shape[0]


                    # Calculate average loss over epoch
                    train_loss = train_loss/data_masks[train_indices].shape[0]
                    train_bce_loss = train_bce_loss / data_masks[train_indices].shape[0]
                    train_dsc_loss = train_dsc_loss / data_masks[train_indices].shape[0]
                    train_loc_loss = train_loc_loss / data_masks[train_indices].shape[0]
                    train_rad_loss = train_rad_loss / data_masks[train_indices].shape[0]
                    train_dice = train_dice/data_masks[train_indices].shape[0]

                    # Get results-flair on the validation and test sets
                    net.eval()
                    with torch.set_grad_enabled(False):
                        # Validation
                        validation_true = []
                        validation_estimated = []
                        val_loss = 0
                        val_bce_loss = 0
                        val_dsc_loss = 0
                        val_loc_loss = 0
                        val_rad_loss = 0
                        val_dice = 0
                        for batch_idx, (idx) in enumerate(validation_dataloader):
                            # Put data on GPU
                            inputs = data_flair_images[idx].to(device)
                            label_mask = data_masks[idx].to(device)
                            label_location = data_location[idx].to(device)
                            # label_radiomics = data_radiomics_temp[idx].to(device)
                            output_seg, output_loc, output_rad = net(inputs)
                            bce_loss, dsc_loss = criterion_seg(output_seg, label_mask)
                            loc_loss = criterion_loc(torch.sigmoid(output_loc), label_location)
                            # rad_loss = criterion_rad(output_rad, label_radiomics)
                            # rad_loss = nn.functional.huber_loss(output_rad, label_radiomics, delta=0.5)
                            loss = bce_loss + dsc_loss + loc_loss #+ rad_loss
                            val_loss += loss.item()*inputs.shape[0]
                            val_bce_loss += bce_loss.item() * inputs.shape[0]
                            val_dsc_loss += dsc_loss.item() * inputs.shape[0]
                            val_loc_loss += loc_loss.item() * inputs.shape[0]
                            # val_rad_loss += rad_loss.item() * inputs.shape[0]
                            val_dice += (1-dsc_loss.item()) * inputs.shape[0]
                        val_loss = val_loss / data_masks[val_indices].shape[0]
                        val_bce_loss = val_bce_loss / data_masks[val_indices].shape[0]
                        val_dsc_loss = val_dsc_loss / data_masks[val_indices].shape[0]
                        val_loc_loss = val_loc_loss / data_masks[val_indices].shape[0]
                        val_rad_loss = val_rad_loss / data_masks[val_indices].shape[0]
                        val_dice = val_dice / data_masks[val_indices].shape[0]

                        # Test
                        test_true_location = []
                        test_estimated_location = []
                        test_true_radiomics = []
                        test_estimated_radiomics = []
                        test_true_segmentation = []
                        test_estimated_segmentation =[]
                        test_loss = 0
                        test_bce_loss = 0
                        test_dsc_loss = 0
                        test_loc_loss = 0
                        test_rad_loss = 0
                        test_dice = 0
                        # Put data on GPU
                        for batch_idx, (idx) in enumerate(test_dataloader):
                            # Put data on GPU
                            inputs = data_flair_images[idx].to(device)
                            label_mask = data_masks[idx].to(device)
                            label_location = data_location[idx].to(device)
                            # label_radiomics = data_radiomics_temp[idx].to(device)
                            output_seg, output_loc, output_rad = net(inputs)
                            bce_loss, dsc_loss = criterion_seg(output_seg, label_mask)
                            loc_loss = criterion_loc(torch.sigmoid(output_loc), label_location)
                            # rad_loss = criterion_rad(output_rad, label_radiomics)
                            # rad_loss = nn.functional.huber_loss(output_rad, label_radiomics, delta=0.5)
                            loss = bce_loss + dsc_loss + loc_loss #+ rad_loss
                            test_loss += loss.item()*inputs.shape[0]
                            test_bce_loss += bce_loss.item() * inputs.shape[0]
                            test_dsc_loss += dsc_loss.item() * inputs.shape[0]
                            test_loc_loss += loc_loss.item() * inputs.shape[0]
                            #test_rad_loss += rad_loss.item() * inputs.shape[0]
                            test_dice += (1-dsc_loss.item()) * inputs.shape[0]
                            test_true_location+=label_location.cpu().detach().numpy().tolist()
                            test_estimated_location+=torch.sigmoid(output_loc).cpu().detach().numpy().tolist()
                            # test_true_radiomics +=label_radiomics.cpu().detach().numpy().tolist()
                            test_estimated_radiomics += output_rad.cpu().detach().numpy().tolist()
                            test_true_segmentation +=label_mask.cpu().detach().numpy().tolist()
                            test_estimated_segmentation += output_seg.cpu().detach().numpy().tolist()
                        test_dice = test_dice / data_masks[test_indices].shape[0]
                        test_loss = test_loss / data_masks[test_indices].shape[0]
                        test_bce_loss = test_bce_loss / data_masks[test_indices].shape[0]
                        test_dsc_loss = test_dsc_loss / data_masks[test_indices].shape[0]
                        test_loc_loss = test_loc_loss / data_masks[test_indices].shape[0]
                        test_rad_loss = test_rad_loss / data_masks[test_indices].shape[0]
                        test_auc = roc_auc_score(test_true_location, test_estimated_location)

                    if val_loss< seg_lowest_val_loss_overall:
                        seg_lowest_val_loss_overall = val_loss
                        seg_lowest_val_loss_epoch = epoch
                        seg_best_test_dice = test_dice
                        seg_best_val_dice = val_dice
                        seg_best_train_dice = train_dice
                        seg_best_test_auc = test_auc
                        seg_best_model_state_dict = net.state_dict()
                        seg_lowest_val_loss_hyper_config = hyper_counter
                        seg_best_lr = seg_lr
                        seg_best_fc_dropout_rate = seg_fc_dropout_rate
                        seg_best_conv_dropout_rate = seg_conv_dropout_rate
                        seg_best_noise = seg_noise_factor
                        seg_best_cosine_length = seg_cosine_length
                        seg_best_global_drop_path_rate = seg_global_drop_path_rate
                        seg_best_local_drop_path_rate = seg_local_drop_path_rate
                        seg_best_do_res = seg_do_res
                        seg_best_decay_rate = seg_weight_decay
                        seg_best_warmup_epoch = seg_warmup_epoch
                        seg_best_radiomics_pred = test_estimated_radiomics
                        seg_best_radiomics_label = test_true_radiomics
                        seg_best_location_pred = test_estimated_location
                        seg_best_location_label = test_true_location
                        seg_best_segmentation_pred = test_estimated_segmentation
                        seg_best_segmentation_label = test_true_segmentation


                    if early_stop:
                        if val_loss < lowest_val_loss_this_hyper:
                            lowest_val_loss_this_hyper = val_loss
                            lowest_val_loss_epoch_this_hyper = epoch


                    epoch_result_string = f"trial: {t}, epoch: {epoch}, " \
                                          f"training loss {round(train_loss,3)} ({round(train_dsc_loss,3)}+{round(train_bce_loss,3)}+{round(train_loc_loss,3)}+{round(train_rad_loss,3)}), " \
                                          f"validation loss: {round(val_loss,3)} ({round(val_dsc_loss,3)}+{round(val_bce_loss,3)}+{round(val_loc_loss,3)}+{round(val_rad_loss,3)}), " \
                                          f"test loss: {round(test_loss, 3)} ({round(test_dsc_loss, 3)}+{round(test_bce_loss, 3)}+{round(test_loc_loss, 3)}+{round(test_rad_loss, 3)}), " \
                                          f"training Dice: {round(train_dice,3)}, validation Dice: {round(val_dice,3)}, " \
                                          f"test Dice: {round(test_dice,3)}, " \
                                          f"test AUC: {round(test_auc, 3)}, " \
                                          f"learning rate: {round(optimizer.param_groups[0]['lr'],5)}, " \
                                          f"Lowest val loss overall: {round(seg_lowest_val_loss_overall, 3)}, on epoch {seg_lowest_val_loss_epoch}," \
                                          f"of hyperparam config {seg_lowest_val_loss_hyper_config},"
                    if early_stop:
                        epoch_result_string+= f", lowest val loss this hyper: {round(lowest_val_loss_this_hyper, 3)}, on epoch {lowest_val_loss_epoch_this_hyper}"
                    print(epoch_result_string)

                    if early_stop:
                        if epoch-lowest_val_loss_epoch_this_hyper>early_stop:
                            print("Stopped early!")
                            break

                    with warmup_scheduler.dampening():
                        scheduler.step()

                torch.cuda.synchronize()
                print("Finished training segmentation model")
                print(f"Time for segmentation: {round(time.time()-track_time_start,3)}")

            seg_validation_dices.append(round(seg_best_val_dice, 3))
            seg_test_dices.append(round(seg_best_test_dice, 3))
            seg_train_dices.append(round(seg_best_train_dice, 3))
            seg_test_aucs.append(round(seg_best_test_auc,3))
            seg_best_epochs.append(round(seg_lowest_val_loss_epoch, 3))
            seg_best_lrs.append(seg_best_lr)
            seg_best_fc_dropout_rates.append(seg_best_fc_dropout_rate)
            seg_best_conv_dropout_rates.append(seg_best_conv_dropout_rate)
            seg_best_noises.append(seg_best_noise)
            seg_best_cosine_lengths.append(seg_best_cosine_length)
            seg_best_global_drop_path_rates.append(seg_best_global_drop_path_rate)
            seg_best_local_drop_path_rates.append(seg_best_local_drop_path_rate)
            seg_best_do_ress.append(seg_best_do_res)
            seg_best_decay_rates.append(seg_best_decay_rate)
            seg_best_warmup_epochs.append(seg_best_warmup_epoch)
            seg_best_radiomics_preds.append(seg_best_radiomics_pred)
            seg_best_radiomics_labels.append(seg_best_radiomics_label)
            seg_best_location_preds.append(seg_best_location_pred)
            seg_best_location_labels.append(seg_best_location_label)


            del net, val_loss, inputs, train_loss, loss, optimizer, noise
            torch.cuda.empty_cache()
            print("Deleted segmentation stuff")


        ##########################################Train the Classification Model##########################################
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_class, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size_class, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_class, shuffle=False)

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
            if pretrain_seg:
                net_dict = net.state_dict()
                pretrain_dict = {k: v for k, v in seg_best_model_state_dict.items() if k in net_dict.keys()}
                # print(pretrain_dict)
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
            print(num_patients_total)
            print(loss_weights)
            loss_weights = loss_weights/torch.sum(loss_weights)
            print(loss_weights)
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
                    # print(loss)
                    # print(label)
                    # print(output)

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
                    val_loss = val_loss / data_masks[val_indices].shape[0]


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

                    if bayesian_inference:
                        net.train()
                        bayesian_preds = []
                        bayesian_labels = []
                        for i in range(10):
                            test_true = []
                            test_estimated = []
                            # Put data on GPU
                            with torch.no_grad():
                                for batch_idx, (idx) in enumerate(test_dataloader):
                                    inputs, label = data_flair_images[idx].to(device), data_labels[idx].to(device)
                                    output = net(inputs)
                                    for i in range(len(label.tolist())):
                                        test_true.append(label[i].cpu().detach().numpy().item())
                                        test_estimated.append(softmax(output[i]).cpu().detach().numpy().tolist())
                                    optimizer.zero_grad()
                                    torch.cuda.empty_cache()
                                bayesian_preds.append(test_estimated)
                                bayesian_labels.append(test_true)







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
        if bayesian_inference:
            bayesian_preds_all.append(bayesian_preds)
            bayesian_labels_all.append(bayesian_labels)
            print(len(bayesian_labels_all))
            print(bayesian_labels_all)

        del net, label, output, val_loss, inputs, train_loss, loss, optimizer, noise
        torch.cuda.empty_cache()

        print(f"Time for this trial: {round(time.time() - time_being_trial,3)}")

        trial_time.append(round(time.time() - time_being_trial,3))

        # Save dictionary of predictions
        timestamp = time.strftime("%m%d-%H%M")

        cols = []
        if pretrain_seg:
            cols = cols + ["Seg Train Dices",
                    "Seg Validation Dices",
                    "Seg Test Dices",
                    "Seg Test AUCs",
                    "Seg best epoch",
                    "Seg learning_rate",
                    "Seg fc_dropout_rate",
                    "Seg conv_dropout_rate",
                    "Seg noise",
                    "Seg cosine length",
                    "Seg global drop path rate",
                    "Seg local drop path rate",
                    "Seg do res",
                    "Seg weight decay",
                    "Seg warmup epochs"]
        if pretrain_rad:
            cols = cols + [
                    "Reg Train Loss",
                    "Reg Validation Loss",
                    "Reg Test Loss",
                    "Reg best epoch",
                    "Reg learning_rate",
                    "Reg fc_dropout_rate",
                    "Reg conv_dropout_rate",
                    "Reg noise",
                    "Reg cosine length",
                    "Reg local drop path rate",
                    "Reg do res",
                    "Reg weight decay",
                    "Reg warmup epochs"]
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
            if pretrain_seg:
                result.append(seg_train_dices[i])
                result.append(seg_validation_dices[i])
                result.append(seg_test_dices[i])
                result.append(seg_test_aucs[i])
                result.append(seg_best_epochs[i])
                result.append(seg_best_lrs[i])
                result.append(seg_best_fc_dropout_rates[i])
                result.append(seg_best_conv_dropout_rates[i])
                result.append(seg_best_noises[i])
                result.append(seg_best_cosine_lengths[i])
                result.append(seg_best_global_drop_path_rates[i])
                result.append(seg_best_local_drop_path_rates[i])
                result.append(seg_best_do_ress[i])
                result.append(seg_best_decay_rates[i])
                result.append(seg_best_warmup_epochs[i])
            if pretrain_rad:
                result.append(reg_train_loss[i])
                result.append(reg_validation_loss[i])
                result.append(reg_test_loss[i])
                result.append(reg_best_epochs[i])
                result.append(reg_best_lrs[i])
                result.append(reg_best_fc_dropout_rates[i])
                result.append(reg_best_conv_dropout_rates[i])
                result.append(reg_best_noises[i])
                result.append(reg_best_cosine_lengths[i])
                result.append(reg_best_local_drop_path_rates[i])
                result.append(reg_best_do_ress[i])
                result.append(reg_best_decay_rates[i])
                result.append(reg_best_warmup_epochs[i])
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
            if bayesian_inference:
                os.remove("bayesian_preds_" + name)
                print(f'Removed file {"bayesian_preds_" + name}')
        except:
            print("No results file to remove")

        try:
            os.remove("preds_" + name)
            print(f"Removed file {'preds_' + name}")
        except:
            print("No predictions file to remove")

        if save_segmentations:
            if t>0:
                with open('segs_true_'+name[:-3] +".npy", 'rb') as f:
                    last_segs_true = np.load(f)
                with open('segs_pred_'+name[:-3] +".npy", 'rb') as f:
                    last_segs_pred = np.load(f)
                os.remove('segs_true_'+name[:-3]+".npy")
                print(f"Removed file {'segs_true_'+name[:-3]+'.npy'}")
                os.remove('segs_pred_'+name[:-3]+".npy")
                print(f"Removed file {'segs_pred_'+name[:-3]+'.npy'}")

        name = ""

        if pretrain_seg:
            name = name + "_segdice" + str(np.round(np.mean(seg_test_dices), 3))
        if pretrain_rad:
            name = name +"_regloss" +str(np.round(np.mean(reg_test_loss),3))

        name = name + "_classauc" + str(np.round(np.mean(class_test_AUCs), 3)) + "_classacc" +str(np.round(np.mean(class_test_accs),3)) +\
               "_npts" +str(len(dataset))  + \
               "_" + timestamp + ".csv"


        name = "plgg_manual_segmentation" + scaling_type + "_seg"  + str(pretrain_seg) + "_rad"  + str(pretrain_rad)\
               + "_results" + "_t" + str(t+1) + name
        print(name)
        results.to_csv(name)

        if save_predictions:
            data = {
                "location_true":seg_best_location_labels  ,
                "location_estimated":seg_best_location_preds,
                "radiomics_true":seg_best_radiomics_labels,
                "radiomics_estimated":seg_best_radiomics_preds,
                # "segmentation_true":seg_best_segmentation_labels,
                # "segmentation_estimated":seg_best_segmentation_preds
            }
            df_predictions = pd.DataFrame(data)
            print(df_predictions)
            df_predictions.to_csv("preds_"+name)

        if bayesian_inference:
            data = {
                "bayesian_preds": bayesian_preds_all,
                "bayesian_labels": bayesian_labels_all,
            }
            df_predictions = pd.DataFrame(data)
            print(df_predictions)
            df_predictions.to_csv("bayesian_preds_" + name)

        if save_segmentations:
            seg_best_segmentation_label=np.array(seg_best_segmentation_label)
            seg_best_segmentation_pred=np.array(seg_best_segmentation_pred)
            if t>0:
                print(last_segs_pred.shape)
                print(last_segs_true.shape)
                seg_best_segmentation_pred = np.concatenate((last_segs_pred, seg_best_segmentation_pred))
                seg_best_segmentation_label = np.concatenate((last_segs_true, seg_best_segmentation_label))
            print(seg_best_segmentation_pred.shape)
            print(seg_best_segmentation_label.shape)
            with open('segs_true_'+name[:-3]+".npy", 'wb') as f:
                np.save(f, seg_best_segmentation_label)
            with open('segs_pred_'+name[:-3]+".npy", 'wb') as f:
                np.save(f, seg_best_segmentation_pred)



