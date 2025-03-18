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
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from loss import SoftDiceBCEWithLogitsLoss
from torch import Tensor
import pytorch_warmup as warmup
from TransBTS_downsample8x_skipconnection import BTS, TransBTS

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

    # location = location_labels[patient_num]
    # location = torch.tensor(location).float()
    #
    # # #Radiomics
    # radiomics = df_rad.loc[patient_num].values
    # radiomics = torch.FloatTensor(radiomics)

    # Load the segmentation data
    mask = np.load(os.path.join(data_dir, str(patient_num), "FLAIR", "preprocessed_segmentation.npy"))
    mask = torch.tensor(mask).float().unsqueeze(0)

    flair_image = np.load((os.path.join(data_dir, str(patient_num), "FLAIR", "preprocessed_FLAIR.npy")))
    flair_image = np.divide(flair_image - np.amin(flair_image), np.amax(flair_image) - np.amin(flair_image))
    flair_input = torch.tensor(flair_image).float().unsqueeze(0)


    # Return the result
    result = {
        "flair_input": flair_input,
        "label": label,
        "mask":mask,
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
    batch_size_seg = 1
    dev_mode = False  # if true, use just a small number of patients, to quickly run through and make sure everything is working as expected
    early_stop = 10
    save_segmentations = False



    seg_param_grid = ParameterGrid([
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


    if dev_mode == True:
        num_trials= 10
        seg_param_grid = ParameterGrid([
            {
                # Segmentation hyperparameters
                "lrs": [0.001, 10],
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

    ############################################Data Preprocessing######################################################
    # Pointing the excel file which contain the data labels
    if socket.gethostname()=='RT6248W-NMH':
        # data_SK_input = pd.read_csv(
        #     r"Z:/Datasets/MedicalImages/BrainData/SickKids/pLGG_EN_Nov2023/pLGG_4cohorts_532subs.csv")
        data_SK_input = pd.read_csv("pLGG_4cohorts_532subs.csv")
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

    # #Radiomics features
    # df_rad = pd.read_csv("Radiomics_binWidth-25_NoNormalization_Whole-Tumor_flair.csv")
    # print(df_rad.shape)
    # df_rad = df_rad.drop(df_rad.columns[[0]+[x for x in range(2,29)]], axis =1)
    # df_rad.set_index('Patient_ID', inplace=True)
    # df_rad = df_rad.filter(like='original', axis=1)
    # # dict_rad_features = {}
    # print(df_rad.shape)

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
                # data_radiomics = torch.unsqueeze(result["radiomics"], 0)
                # data_location = torch.unsqueeze(result["location"], 0)
            else:
                data_labels = torch.cat((data_labels, torch.unsqueeze(result["label"], 0)))
                data_masks = torch.cat((data_masks, torch.unsqueeze(result["mask"], 0)))
                data_flair_images = torch.cat((data_flair_images,torch.unsqueeze(result["flair_input"],0)))
                # data_radiomics = torch.cat((data_radiomics, torch.unsqueeze(result["radiomics"], 0)))
                # data_location = torch.cat((data_location, torch.unsqueeze(result["location"], 0)))
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

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_seg, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size_seg, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_seg, shuffle=True)

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
                seg_epochs = 10

            # Define neural network stuff
            track_time_start = time.time()

            # net = BTS(img_dim=128,
            #         patch_dim=8,
            #         num_channels=1,
            #         num_classes=1,
            #         embedding_dim=512,
            #         num_heads=8,
            #         num_layers=1,
            #         hidden_dim=512,
            #         dropout_rate=0.1,
            #         attn_dropout_rate=0.1,
            #         conv_patch_representation=True,
            #         positional_encoding_type="learned",
            #     )

            _, net = TransBTS()
            net.to(device)
            trainable_params = sum(
                p.numel() for p in net.parameters() if p.requires_grad
            )
            print(trainable_params)


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

                    # Add noise to images
                    noise = torch.randn_like(inputs, device=device)*seg_noise_factor
                    inputs = inputs+noise
                    # Forward + Backward + Optimize
                    output_seg = net(inputs)
                    bce_loss, dsc_loss = criterion_seg(output_seg, label_mask)
                    # rad_loss = nn.functional.huber_loss(output_rad, label_radiomics, delta=0.5)
                    loss = bce_loss + dsc_loss
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
                        output_seg  = net(inputs)
                        bce_loss, dsc_loss = criterion_seg(output_seg, label_mask)
                        # rad_loss = nn.functional.huber_loss(output_rad, label_radiomics, delta=0.5)
                        loss = bce_loss + dsc_loss
                        val_loss += loss.item()*inputs.shape[0]
                        val_bce_loss += bce_loss.item() * inputs.shape[0]
                        val_dsc_loss += dsc_loss.item() * inputs.shape[0]
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
                        output_seg = net(inputs)
                        bce_loss, dsc_loss = criterion_seg(output_seg, label_mask)
                        # rad_loss = nn.functional.huber_loss(output_rad, label_radiomics, delta=0.5)
                        loss = bce_loss + dsc_loss
                        test_loss += loss.item()*inputs.shape[0]
                        test_bce_loss += bce_loss.item() * inputs.shape[0]
                        test_dsc_loss += dsc_loss.item() * inputs.shape[0]
                        test_dice += (1-dsc_loss.item()) * inputs.shape[0]
                        # test_true_segmentation +=label_mask.cpu().detach().numpy().tolist()
                        # test_estimated_segmentation += output_seg.cpu().detach().numpy().tolist()
                    test_dice = test_dice / data_masks[test_indices].shape[0]
                    test_loss = test_loss / data_masks[test_indices].shape[0]
                    test_bce_loss = test_bce_loss / data_masks[test_indices].shape[0]
                    test_dsc_loss = test_dsc_loss / data_masks[test_indices].shape[0]

                if val_loss< seg_lowest_val_loss_overall:
                    seg_lowest_val_loss_overall = val_loss
                    seg_lowest_val_loss_epoch = epoch
                    seg_best_test_dice = test_dice
                    seg_best_val_dice = val_dice
                    seg_best_train_dice = train_dice
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
                    # seg_best_segmentation_pred = test_estimated_segmentation
                    # seg_best_segmentation_label = test_true_segmentation


                if early_stop:
                    if val_loss < lowest_val_loss_this_hyper:
                        lowest_val_loss_this_hyper = val_loss
                        lowest_val_loss_epoch_this_hyper = epoch


                epoch_result_string = f"trial: {t}, epoch: {epoch}, " \
                                      f"training loss {round(train_loss,3)} ({round(train_dsc_loss,3)}+{round(train_bce_loss,3)}), " \
                                      f"validation loss: {round(val_loss,3)} ({round(val_dsc_loss,3)}+{round(val_bce_loss,3)}), " \
                                      f"test loss: {round(test_loss, 3)} ({round(test_dsc_loss, 3)}+{round(test_bce_loss, 3)}), " \
                                      f"training Dice: {round(train_dice,3)}, validation Dice: {round(val_dice,3)}, " \
                                      f"test Dice: {round(test_dice,3)}, " \
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
        # seg_test_aucs.append(round(seg_best_test_auc,3))
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



        del net, val_loss, inputs, train_loss, loss, optimizer, noise
        torch.cuda.empty_cache()
        print("Deleted segmentation stuff")

        print(f"Time for this trial: {round(time.time() - time_being_trial,3)}")

        trial_time.append(round(time.time() - time_being_trial,3))

        # Save dictionary of predictions
        timestamp = time.strftime("%m%d-%H%M")

        cols = []

        cols = cols + ["Seg Train Dices",
                    "Seg Validation Dices",
                    "Seg Test Dices",
                    # "Seg Test AUCs",
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
                    "Seg warmup epochs",
                    "Times"]

        results = pd.DataFrame(
            columns=[cols])

        print(results)
        for i in range(len(seg_train_dices)):
            print(i)
            result = []

            result.append(seg_train_dices[i])
            result.append(seg_validation_dices[i])
            result.append(seg_test_dices[i])
            # result.append(seg_test_aucs[i])
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



        name = "segdice" + str(np.round(np.mean(seg_test_dices), 3))
        name = name +\
               "_npts" +str(len(dataset))  + \
               "_" + timestamp + ".csv"


        name = "transbts_plgg_results_t" + str(t+1) + name
        print(name)
        results.to_csv(name)

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



