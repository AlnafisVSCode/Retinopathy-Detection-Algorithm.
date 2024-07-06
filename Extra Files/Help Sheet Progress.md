
**Preparing the data**

**datasetup.py** 
- I changed the structure of the images to fit with PyTorch's Dataloader 
	- Originally all the images were in the directory ./traindl
	- I created a folder ./train and then created a subfolder with the name of the label assigned to the image (found in the .csv)
	- Then I moved all the images into ./train/{label}/ corresponding to the correct label

**imageSizes.py**
(Purpose) to find statistics about the dataset and remove small images

- I looked at the images that I had to work with and noticed the images were cropped at different parts and the images were taken from different angles and rotations.
- I made this script to print the mean width,height of the images in the dataset which I then used to determine the minimum size of the images I would like to train with.
- I also displayed how many images would be above my desired size per label from the dataset as I wanted to remove the outliers without reducing the size of my dataset too much.
- The script then deleted a small portion of images which were below 1920x1080

**Sizing.py**
(Purpose) To figure out the mean aspect ratio , Percent of aspect ratio outliers.

- I calculated & printed the Mean Aspect ratio (by label) , Mean overall Aspect ratio & number of outliers and saw that only <1% of the dataset was +- 20% of the 16:9 aspect ratio and <20% of the dataset was += 10% of the 16:9 aspect ratio.

- Most of the dataset fit into this 16:9 aspect ratio so I then moved on to the next step.
 I resized all larger images to 1920x1080 while keeping the same 16:9 aspect ratio by adding a black padding before resizing with PIL.
	- The background of the images were all black which is why I used a black padding, this would then not interfere with the data in the images.

**Preprocessing**
- I needed to get values for my Standard deviation & Mean to help normalize the data during training. I wanted to load the entire dataset into memory but my computer would not be able to handle 35GB of images at once therefore I came up with two different ideas to calculate Standard deviation and mean RGB Pixel values.


**Preprocessing.py**
- Method #1
- I filled the sum_values list with the sum of the pixel values for the 3 channels
- I filled the sum_squares list with the sum squared pixel values for the 3 channels
- I then calculated the mean of the pixel values for the full dataset
- I then calculated the standard deviation of the pixel values for the full dataset.
This method allowed me to store the integer values of the pixels in memory while only storing 8 images at a time in memory to calculate an accurate value of the Standard deviation & Mean while using a low memory usage alternative.

I then printed these values so I could use them as my final transform while training to normalize the data.

**Preprocessing2.py**
- Method #2
- I decided to split up the channels further and calculate the Mean and standard deviation per channel for R,G,B 
- I also stored an integer value as the total for the sum & sum_sq R,G,B values rather than a list.
- I predicted this method would be more accurate as it explicitly separates the R,G,B channels for the Standard deviation and mean calculations .

During training I used both STD,Mean Values as normalization to compare which one had the best valuation Accuracy and lowest valuation Loss.



**Training**

- I decided to use Transfer learning and use a pre-trained CNN called ResNet for classifying images. ResNet was trained using the ImageNet dataset which used images of size 224x224.

- I began training with a ResNet18 Model and decided to first try the dimensions 256,144 (Keeping the 16:9 Aspect ratio)
	- I picked this because the ImageNet used images of size 224x224 
- The accuracy did not improve and the loss did not decrease.
	- The same issue recurred on both STD,Mean value calcuator methods.


- Next I decided to try the ResNet34 Model using the following parameters:
Learning Rate = 0.045
Batch_size=32
momentum = 0.9

At Epoch 1:  
train Loss: 0.9559 Acc: 0.7302                                                                                          
val Loss: 0.8637 Acc: 0.7282

Epoch 20/50                                                                                       
train Loss: 0.8244 Acc: 0.7347                                                                                          
val Loss: 0.8517 Acc: 0.7265

I adjusted the learning rate to 0.025 and tried again, but after 3 epochs I noticed it still was not training so I moved on.


- Next I decided to try the same task but with ResNet50 because it was a deeper network that can find more complex patterns. 

**Params used:**
Learning Rate = 0.045
Batch_size=32
momentum = 0.9

After 20 Epoch's I had
92.55% Training Accuracy, 0.2 Training Loss
72% Evaluation Accuracy, 1.26 Evaluation Loss

The Evaluation Accuracy slowly decreased & Loss gradually increased each epoch which were signs of overfitting.

I added a weight decay to combat this issue.
Weight decay = 0.005

I also added more Transforms:
RandomVerticalFlip() - because the images could be inverted.
RandomRotation(10) - minor rotations can simulate variable capturing angles.
ColorJitter - Due to the different colour of images
RandomAffine() - I also added some slight translations in axis for the images.

& then began training again.

After 8 Epoch's I noticed the issue was recurring, Training Loss and accuracy improving drastically, Val Loss & Accuracy only decreasing. This looked like clear signs of overfitting yet again.

I believe this was caused by there not being enough pixels available for the model to learn. 

This time I increased the Resize transform to (200, 200) which is half the original resolution of 1920x1080, I also had to decrease the batch size to 8 for the images to fit into memory.

It started training but but after 8 epoch's overfitting became a problem again and the Evaluation Accuracy started to drop along with the Evaluation Loss increasing.

I increased the weight decay from 0.005 to 0.05 

I ran for 10 Epoch's however there was no improvement in learning, I believe the weight decay had been too high

**tuner.py**
- I created a script with the purpose of testing different weight decay's to find an optimal value to prevent overfitting but while learning the most effectively with the dataset.
- I have included a tuner_results.txt containing the output of the result values
- The tuner script runs 3 epoch's because every epoch takes a very long time and it would be just enough to see if the model was learning.
- I discovered that using the weight decay 0.01 had the lowest Evaluation loss


I tried to train the model further with the weight decay set to 0.01 however it did not improve the Evaluation Accuracy and the Evaluation loss continued to grow.


- Next I tried adjusting the learning rate &
I reduced the weight decay to 0.00025
& tested different learning rates:

I tested 4 different learning rates from 0.02, 0.015, 0.01, 0.075

and found with the learning rate 0.01:
after 40 Epoch's the results were:

train Loss: 0.5276 Acc: 82.47%
val Loss: 0.5556 Acc: 82.6% 
