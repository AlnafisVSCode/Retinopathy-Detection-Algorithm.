# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets , models , transforms
import time
import os
import copy
from termcolor import cprint
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt
torch.manual_seed(42)

# Check if CUDA is available and set PyTorch to use CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initiated on device: {device}")

# Define a custom dataset that also returns the image file path in addition to the image and label
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


#define a class for the model
class Detection:
	# Define the class labels
	LABEL_NAMES = {
		0 : "No DR",
		1: "Mild DR",
		2: "Moderate DR",
		3: "Severe DR",
		4: "Proliferative DR"
		}

	    # Method to denormalize image tensors
	def denormalize (self,tensor):
		for t , m , s in zip(tensor , self.mean , self.std):
			t.mul_(s).add_(m)
		return tensor

	# Initialize the model
	def __init__ (self , data_dir , save_path , learning_rate, weight_decay, save, gamma=0.9):

		# Initialize properties
		self.data_dir = data_dir
		self.learning_rate = learning_rate
		self.weightDecay=weight_decay
		self.save=save
		self.gamma = gamma

		self.val_loss = 100
		self.val_accuracy = 0

		self.validationAccuracy = 0

		self.save_path = save_path
		self.mean = np.array([ 0.2194 , 0.1533 , 0.1099 ])
		self.std = np.array([ 0.2889 , 0.2069 , 0.1613 ])
		self.dataloaders , self.dataset_sizes , self.class_names = self.prepare_data()
		self.model , self.optimizer , self.epoch_start = self.load_state(train = True)


    # Define a method to get the validation accuracy
	def get_validation_accuracy(self): return self.validationAccuracy

	def prepare_data (self):
		# Define the data transforms for training and validation sets
		data_transforms = {
			'train': transforms.Compose([
				transforms.Resize((200 , 200)) ,
				transforms.RandomHorizontalFlip() ,
				transforms.RandomVerticalFlip() ,
				transforms.RandomRotation(10) ,
				transforms.ColorJitter(brightness = 0.2 , contrast = 0.2 , saturation = 0.2) ,
				transforms.RandomAffine(degrees = 0 , translate = (0.1 , 0.1)) ,
				transforms.ToTensor() ,
				transforms.Normalize(self.mean , self.std)
				]) ,
			'val': transforms.Compose([
				transforms.Resize((200 , 200)) ,
				transforms.ToTensor() ,
				transforms.Normalize(self.mean , self.std)
				]) ,
			}
		# Define the datasets and dataloaders
		datasets_dict = {
			'train': datasets.ImageFolder(os.path.join(self.data_dir , 'train') , transform = data_transforms[ 'train' ]) ,
			'val': ImageFolderWithPaths(os.path.join(self.data_dir , 'test'))
			}
		# Split the validation set into a test and validation set
		datasets_dict[ 'val' ].transform = data_transforms[ 'val' ]
		dataloaders = {
			x: torch.utils.data.DataLoader(datasets_dict[ x ] , batch_size = 32 , shuffle = True , num_workers = 12) for
			x in [ 'train' , 'val' ]
			}
		
		# Get the dataset sizes and class names
		dataset_sizes = { x: len(datasets_dict[ x ]) for x in [ 'train' , 'val' ] }
		class_names = datasets_dict[ 'train' ].classes

		return dataloaders , dataset_sizes , class_names

	# Define a method to load the model state
	def load_state (self , train = False):
		model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs , len(self.class_names))

		optimizer = optim.SGD(model.parameters() , lr = self.learning_rate , momentum = 0.9, weight_decay = self.weightDecay)

		epoch_start = 0
		if os.path.exists(self.save_path):
			checkpoint = torch.load(self.save_path)
			model.load_state_dict(checkpoint[ 'model_state' ])
			optimizer.load_state_dict(checkpoint[ 'optimizer_state' ])
			for state in optimizer.state.values():
				for k , v in state.items():
					if isinstance(v , torch.Tensor):
						state[ k ] = v.to(device)
			epoch_start = checkpoint[ 'epoch' ]

			self.val_loss = checkpoint[ 'val_loss' ]
			self.val_accuracy = checkpoint[ 'val_accuracy' ]


			if train:
				model.train()
				cprint("Loaded model Mode: Training!" , "green")
			else:
				model.eval()
				cprint("Loaded model Mode: Evaluation!" , "green")

		model = model.to(device)
		return model , optimizer , epoch_start

	# Define a method to save the model state
	def save_state (self , epoch , model , optimizer):
		checkpoint = {
			'model_state': model.state_dict() ,
			'optimizer_state': optimizer.state_dict() ,
			'epoch': epoch,
			'val_loss' : self.val_loss,
			'val_accuracy': self.val_accuracy
			}
		torch.save(checkpoint , self.save_path)

	# Method to train the model
	def train (self , num_epochs = 25):
		since = time.time()

		best_model_wts = copy.deepcopy(self.model.state_dict())
		best_acc = 0.0
		# Define the loss function and scheduler (for updating the learning rate)
		criterion = nn.CrossEntropyLoss()
		scheduler = lr_scheduler.StepLR(self.optimizer , step_size = 5 , gamma = self.gamma)

		for epoch in range(num_epochs):
			cprint('-' * 30 , "cyan")
			cprint(f'Epoch {epoch + 1}/{num_epochs}' , "green")


			for phase in [ 'train' ]:
				if phase == 'train':
					self.model.train()
				else:
					self.model.eval()

				running_loss = 0.0
				running_corrects = 0
				# Loop over the data
				for inputs , labels in self.dataloaders[ phase ]:
					inputs = inputs.to(device)
					labels = labels.to(device)

					self.optimizer.zero_grad()
					# Forward pass
					with torch.set_grad_enabled(phase == 'train'):
						outputs = self.model(inputs)
						_ , preds = torch.max(outputs , 1)
						loss = criterion(outputs , labels)

						if phase == 'train':
							loss.backward()
							self.optimizer.step()

					# Statistics
					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels.data)
                # Calculate epoch loss and accuracy
				epoch_loss = running_loss / self.dataset_sizes[ phase ]
				epoch_acc = running_corrects.double() / self.dataset_sizes[ phase ]

				cprint(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}' , "green")

                # Save the model if it's the best one so far
				if phase == 'val' and epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(self.model.state_dict())
			scheduler.step()

            # Save the model state if necessary
			if self.save:
				loss , acc = self.evaluate()
				if loss <= self.val_loss and self.val_accuracy <= self.val_accuracy:
					self.save_state(self.epoch_start + num_epochs , self.model , self.optimizer)

        # Print out training time and best validation accuracy
		time_elapsed = time.time() - since
		cprint('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60) , 'green')
		cprint('Best val Acc: {:4f}'.format(best_acc) , 'green')

		self.model.load_state_dict(best_model_wts)



    # Define a method to evaluate the model
	def evaluate (self,print_every=False):
		cprint("Loaded model Mode: Evaluation!" , "green")
		self.model.eval()
		phase = "val"
		running_loss = 0.0
		running_corrects = 0
		correct = 0
		incorrect = 0
		criterion = nn.CrossEntropyLoss()

		for inputs , labels , filenames  in self.dataloaders[ phase ]:


			inputs = inputs.to(device)
			labels = labels.to(device)

			with torch.no_grad():
				outputs = self.model(inputs)
				_ , preds = torch.max(outputs , 1)
				loss = criterion(outputs , labels)

			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)

			for i in range(len(labels)):

				label = labels[ i ].item()
				prediction_label = preds[ i ].item()
				filename = filenames[ i ]
				#print(preds[ i ])

				if label == prediction_label:
					correct += 1
				else:
					incorrect += 1
				if print_every:

					print_output = f"Answer: {label} , AI Prediction: {prediction_label} [ {correct} Correct ] , [{incorrect} Incorrect ]\n{filename} AI Predicted {self.LABEL_NAMES.get(prediction_label)} & Answer was [{self.LABEL_NAMES.get(label)}]\n"
					if prediction_label == label:
						cprint(print_output , "green")
					else:
						cprint(print_output , "red")

		epoch_loss = running_loss / self.dataset_sizes[ phase ]
		epoch_acc = running_corrects.double() / self.dataset_sizes[ phase ]

		cprint(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}' , "green")

		ACCURACY = (correct / (correct + incorrect)) * 100
		print(f"AI Predicted with {ACCURACY:.2f}% Accuracy , {correct} correct & {incorrect} incorrect")
		return epoch_loss,epoch_acc

	def display_results (self , save_as = "results.png"):
		# Ensure model is in evaluation mode
		self.model.eval()

		# Use a dataloader for the test set (assuming it's named "val")
		data_iter = iter(self.dataloaders[ 'val' ])

		# Initialize an empty list to store found samples
		found_samples = { label: [ ] for label in range(5) }

		# Loop over the data until we find at least three samples for each class label (0-4)
		while any(len(samples) < 3 for samples in found_samples.values()):
			inputs , labels , filenames = next(data_iter)

			for i in range(len(labels)):
				label_item = labels[ i ].item()
				if len(found_samples[ label_item ]) < 3:
					found_samples[ label_item ].append((inputs[ i ] , labels[ i ] , filenames[ i ]))

		# Create the 5x3 display grid
		fig , axarr = plt.subplots(5 , 3 , figsize = (9 , 15))

		for label in range(5):
			for idx , (input_img , label , filename) in enumerate(found_samples[ label ]):
				# Move tensor to device and get model's prediction
				input_img = input_img.unsqueeze(0).to(device)  # Add batch dimension
				with torch.no_grad():
					outputs = self.model(input_img)
					_ , pred = torch.max(outputs , 1)

				# Denormalize and display the image, AI label, and real label
				image_to_display = self.denormalize(input_img[ 0 ].clone().detach().cpu())
				axarr[ label , idx ].imshow(image_to_display.permute(1 , 2 , 0).numpy())

				axarr[ label , idx ].set_title(
					f"AI Label: {self.LABEL_NAMES.get(pred[ 0 ].item())}\nReal Label: {self.LABEL_NAMES.get(label.item())}"
					)
				axarr[ label , idx ].axis('off')

		# Save and display the final image
		plt.tight_layout()
		plt.savefig(save_as)
		plt.show()

if __name__ == "__main__":
	# Example usage

	MODEL_NAME = "eye_eye_model"
	"""
	LEARNING_RATE = 0.005
	model_detection = Detection(data_dir = './' , save_path = f'eye_{MODEL_NAME}.pth',
								learning_rate=LEARNING_RATE , weight_decay = 0.00025, save=True
								)
	model_detection.train(num_epochs = 20)

	LEARNING_RATE = 0.0025
	model_detection = Detection(data_dir = './' , save_path = f'eye_{MODEL_NAME}.pth',
								learning_rate=LEARNING_RATE , weight_decay = 0.00025, save=True
								)
	model_detection.train(num_epochs = 20)

	LEARNING_RATE = 0.0015
	model_detection = Detection(data_dir = './' , save_path = f'eye_{MODEL_NAME}.pth',
								learning_rate=LEARNING_RATE , weight_decay = 0.00025, save=True
								)
	model_detection.train(num_epochs = 20)"""

	# Accuracy is exceeding the training accuracy so I will try increasing weight decay

	# Lr 0.0075 && Weight decay 0.0025 Caused Overfitting, Big increase in Val Loss, Train Accuracy

	"""
	Epoch 20/20                                                                                                             
	train Loss: 0.2581 Acc: 0.9031                                                                                          
	val Loss: 0.8699 Acc: 0.7541
	"""

	"""
	I ran tuner.py to find good values to further the training:
	
	AI Predicted with 76.63% Accuracy , 5382 correct & 1641 incorrect                                                       
	Evaluation with weight decay 0.5 with Learning rate 0.025 took 0.18 hours and achieved accuracy: 0.7663                 
	Best weight decay found: 0.5 with validation accuracy: 0.7726
	
	"""



	LEARNING_RATE=0.025
	model_detection = Detection(data_dir = './' , save_path = f'{MODEL_NAME}.pth',
									learning_rate=LEARNING_RATE , weight_decay = 0.5, save=True, gamma = 1
								)
	model_detection.train(num_epochs = 1)
	model_detection.evaluate(print_every=True)
	# model_detection.display_results()

