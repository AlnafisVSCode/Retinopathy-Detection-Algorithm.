import numpy as np
import time
from model import Detection

class HyperparameterTuner:

	def __init__ (self , data_dir , save_path , num_epochs_per_run = 3 , num_runs = 10):
		self.data_dir = data_dir
		self.save_path = save_path
		self.num_epochs_per_run = num_epochs_per_run
		self.num_runs = num_runs

	def objective_function (self , weight_decay):
		model_detection = Detection(data_dir = self.data_dir , save_path = self.save_path ,
									learning_rate = 0.045, weight_decay=weight_decay,
									save=False
									)
		model_detection.train(num_epochs = self.num_epochs_per_run)


		# NOTE: You will need a method in the Detection class to get the validation performance
		# For this example, I'll assume you have a method `get_validation_accuracy()` that returns the validation accuracy
		val_accuracy = model_detection.get_validation_accuracy()

		return val_accuracy

	def random_search (self):
		best_weight_decay = None
		best_val_accuracy = float('-inf')

		# Define a range for weight decay. This can be adjusted.
		weight_decay_range = [
			#1e-5 ,
							   1e-4 ,
							   1e-3 ,
							   1e-2 ,
							   0.05 ,
							   0.1 ,
							   0.5 ]

		for wd in weight_decay_range:

			start_time = time.time()

			val_accuracy = self.objective_function(wd)

			if val_accuracy > best_val_accuracy:
				best_val_accuracy = val_accuracy
				best_weight_decay = wd

			elapsed_time = time.time() - start_time
			print(
				f"Evaluation with weight decay {wd} took {elapsed_time / 3600:.2f} hours and achieved accuracy: {val_accuracy:.4f}")

		print(f"Best weight decay found: {best_weight_decay} with validation accuracy: {best_val_accuracy:.4f}")


tuner = HyperparameterTuner(data_dir = 'train' , save_path = 'eye.pth')
tuner.random_search()
