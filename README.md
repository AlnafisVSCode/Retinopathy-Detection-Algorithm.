To execute **Model.py**, ensure you have a CUDA-compatible GPU, as it significantly speeds up the process by running on the GPU. If CUDA is unavailable, the script will default to CPU execution, which may result in slower performance.

Upon launching, you should see a confirmation message indicating the device being used, either "Initiated on device: cuda" for GPUs or "Initiated on device: CPU" for CPU usage. The script will proceed with loading the model in two modes, displaying:

```
"Loaded model Mode: Training!
Loaded model Mode: Evaluation!"
```

As the script runs, it will produce outputs in the following format:

```
e.g., 
Answer: 2, AI Prediction: 0 [5471 Correct], [1545 Incorrect]
./test\2\40676_right.jpeg AI Predicted No DR & Answer was [Moderate DR]

Answer: 0, AI Prediction: 0 [5472 Correct], [1545 Incorrect]
./test\0\6963_left.jpeg AI Predicted No DR & Answer was [No DR]
```

This output details the correct and incorrect predictions made by the AI, along with the specific images assessed and their corresponding diagnosis.

After processing, the script generates a matplotlib diagram showcasing some of the Diabetic Retinopathy (DR) images alongside the AI's predictions and outcomes.

To finalize the process and save the generated images, simply close the matplotlib window. This action triggers the saving of the diagram as an image file, marking the completion of the run.


![results](https://github.com/AlnafisVSCode/Retinopathy-Detection-Algorithm./assets/99893321/62ed5017-e9b9-4fb8-8316-68b7b64c079b)
