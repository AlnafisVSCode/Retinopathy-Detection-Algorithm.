Run Model.py

(CUDA REQUIRED) - runs on GPU to run it faster
It checks for CUDA. Otherwise, it will run on the CPU but might be very slow to finish the output.


Once started it should display: "Initiated on device: (cuda or CPU)" 
"""Loaded model Mode: Training!
Loaded model Mode: Evaluation!"""

then ->

e.g. 
Answer: 2 , AI Prediction: 0 [ 5471 Correct ] , [1545 Incorrect ]
./test\2\40676_right.jpeg AI Predicted No DR & Answer was [Moderate DR]

Answer: 0 , AI Prediction: 0 [ 5472 Correct ] , [1545 Incorrect ]
./test\0\6963_left.jpeg AI Predicted No DR & Answer was [No DR]

after -> 

runs again to produce a matplotlib Diagram displaying some DR images with AI guesses and output.

->
Close the windows of the matplot to save as an image and complete the run.
----------------------------------------------------------------------------------------------------------------------

Additional information on Help_sheet_Progress.py in file 'Extra Files'

