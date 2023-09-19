import time
import warnings
import torch
from torch.utils.data import DataLoader
import random
import numpy as np

from atlas import atlas_light
from datafuncs import DriveDataset
import trainfuncs

warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.autograd.detect_anomaly()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DATA_PATH = "dataset_50_256/train"
VAL_DATA_PATH = "dataset_50_256/validation"
TEST_DATA_PATH = "dataset_50_256/test"



"""

------------------- MODEL PARAMETERS -------------------

"""


IMG_SIZE = 256
IMG_CHANNELS = 2
LABELS = 8

BATCH_SIZE = 8
EPOCHS = 500

DROP = 0.4
LEARNING_RATE = 1e-4
LR = [1e-3, 4e-4, 2e-4, 1e-4, 5e-5, 1e-5][2:]
LR_SCHEDULE = True

PATIENCE = 5
PRODUCE_PLOTS = False
SHUFFLE = True

SENSORS = 7
ARCHITECTURE = 'Atlas-Light'

NAME = f'{ARCHITECTURE}_{IMG_SIZE}_{BATCH_SIZE}BS_{IMG_CHANNELS}C_{LABELS}L_{DROP}D_{SENSORS}S_{TRAIN_DATA_PATH.split("/")[-2].split("_")[-1]}HZ_{int(time.time())}.pt'

model = atlas_light(drop=DROP, img_size=IMG_SIZE).to(DEVICE)

OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


"""

------------------- TRAINING & TESTING -------------------

"""


MODE = 2  # 1: Train | 2: test

SAVED_MODEL = "Atlas-Light_256_8BS_2C_8L_0.4D_7S_50HZ.pt"


"""

------------------- MAIN SCRIPT START -------------------

"""

train_set = DriveDataset(img_path=TRAIN_DATA_PATH, img_size=IMG_SIZE, img_channels=IMG_CHANNELS, labels=LABELS, set_name="Train")
valid_set = DriveDataset(img_path=VAL_DATA_PATH, img_size=IMG_SIZE, img_channels=IMG_CHANNELS, labels=LABELS, set_name="Validation")

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE)

test_set = DriveDataset(img_path=TEST_DATA_PATH, img_size=IMG_SIZE, img_channels=IMG_CHANNELS, labels=LABELS, set_name="Test")
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=SHUFFLE)


if MODE == 1:
    trainfuncs.train(model, train_loader, valid_loader, EPOCHS, OPTIMIZER, DEVICE, NAME, LR, PATIENCE, LR_SCHEDULE, PRODUCE_PLOTS)
elif MODE == 2:
    model.load_state_dict(torch.load(SAVED_MODEL, map_location=DEVICE)) 
    trainfuncs.test(model, test_loader, DEVICE, SAVED_MODEL)
else:
    print(f"Invalid Mode: {MODE}")