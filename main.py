import sys
from src.train.trainer import Trainer
from src.model.model_factory import build_model
from src.logger.loggers import build_logger
from src.utils.load_config import load_config 
from src.utils.dataset_type import DatasetType 
from src.evalutation.writers import build_writers
from src.evalutation.evaluators import build_evaluator
import torchvision
import torchvision.transforms as transforms
from get_data import UltrasoundImageDataset
import os
import torch.nn as nn

config_file = sys.argv[1]
config = load_config(config_file)

model = build_model(config.model)
logger = build_logger(config)
writers = build_writers(config, config.train.out_path, logger)
train_evaluator = build_evaluator(config.evaluation.train_metrics, writers, DatasetType.Train)
valid_evaluator = build_evaluator(config.evaluation.valid_metrics, writers, DatasetType.Valid)

trainer = Trainer(model, logger, train_evaluator, valid_evaluator, config)

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


import pdb

# Preprocess Images

from src.preproc.preproc import process_images_in_folder

input_folder = 'data/images'
output_folder = 'data/processed_images'
process_images_in_folder(input_folder, output_folder)



transformation = transforms.Compose([
    # transforms.Resize((64, 64)),        # THE IMAGES ARE RE-SCALED TO 64x64
    # transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])


dataset = UltrasoundImageDataset("data/labels_students.csv", "data/processed_images", transform=transformation)

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.1, 0.2])

trainloader = DataLoader(
    train_dataset, 
    batch_size=config.data.batch_size, 
    shuffle=config.data.shuffle
)
validloader = DataLoader(
    validation_dataset, 
    batch_size=config.data.batch_size, 
    shuffle=config.data.shuffle
)
testloader = DataLoader(
    test_dataset, 
    batch_size=config.data.batch_size, 
    shuffle=config.data.shuffle
)


trainer.train(trainloader, validloader)


 # Instantiate your model
model.load_state_dict(torch.load(os.path.join(config.train.out_path, config.name + '_best.pth')))  # Load trained weights

# 2. Set the model to evaluation mode
model.eval()

# 3. Iterate over batches in your test loader
total_loss = 0
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(model.device)  # Assuming you're using GPU, move inputs to device
        labels = labels.to(model.device)  # Move labels to device
        
        # 4. Forward pass each batch through the model
        outputs = model(inputs)
        
        # 5. Compute metrics
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)  # Calculate loss
        total_loss += loss.item()  # Accumulate loss
        
        _, predicted = torch.max(outputs, 1)  # Get predicted labels
        correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
        total_samples += labels.size(0)  # Count total samples

# Calculate average loss and accuracy
average_loss = total_loss / len(testloader)
accuracy = correct_predictions / total_samples

logger.log("Average loss in test:", average_loss)
logger.log("Accuracy: ", accuracy)