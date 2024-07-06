import pandas as pd
import os
import shutil

# Load the CSV file
df = pd.read_csv('trainLabels.csv')

# Define a dictionary to track the existing classes
existing_classes = { }

# Iterate over each row in the DataFrame
for index , row in df.iterrows():
	image_name = row[ 'image' ]
	level = row[ 'level' ]

	# Check if this level has been seen before, if not create a directory for it
	if not existing_classes.get(level):
		existing_classes[ level ] = True
		os.makedirs(f"./train/{level}" , exist_ok = True)

	# Copy the image to the appropriate class directory
	src_path = os.path.join("./traindl" , f"{image_name}.jpeg")
	dest_path = os.path.join("./train" , f"{level}" , f"{image_name}.jpeg")
	shutil.copy(src_path , dest_path)
