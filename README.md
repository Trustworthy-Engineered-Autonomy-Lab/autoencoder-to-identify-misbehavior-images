# autoencoder-to-identify-misbehavior-images

## Autoencoder for Image Reconstruction and Anomaly Detection

- This repository modifies the autoencoder model for image reconstruction and anomaly detection. The primary goal is to train the autoencoder on normal images and use it to identify misbehavior or blurred images by calculating reconstruction errors.


### Features

- **Data Augmentation:** Applies random transformations to training images to improve generalization.
- **Efficient Training:** Trains the autoencoder in batches to handle large datasets without overwhelming system resources.
- **Early Stopping and Checkpoints:** Uses early stopping to prevent overfitting and model checkpoints to save the best model during training.
- **Custom Autoencoder Architecture:** Implements a convolutional autoencoder with dropout layers for regularization.
- **Subset Training:** Trains on subsets of data to handle large datasets efficiently.


### Dependencies

- TensorFlow
- Keras
- NumPy
- tqdm (for progress bars)
- os (for file handling)
- random (for shuffling)


### Data

The cleaned data is used for this purpose, and the entire cleaned data is used to train the model to improve accuracy and efficiency for more generalizability. 

The dataset consists of:
- **9507 normal images** in grayscale format.
- **787 misbehavior (blurred) image**s in grayscale format.

The misbehavior images include 787 images (including images with walls). This clean data does not contain any duplicates.

Both sets of images are resized to 224x224 pixels.

*Ensure the data is correctly placed in the normal_images and misbehavior_images directories. The data is not included in this repository.


### Training the Model

The autoencoder is trained on the entire dataset with 10 subsets of 1000 images in each subset (950 normal and 50 misbehavior) at a time to avoid memory issues. 
The training process involves:

- **Loading Images:** Images are loaded from the specified directories using the load_images function.
- **Data Augmentation:** Applied using ImageDataGenerator to improve the model's ability to generalize.
- **Training in Batches:** The model is trained in batches with early stopping and checkpoint callbacks to save the best model.
- **Subsets:** The dataset is divided into **10 subsets**, each containing 1000 images (950 normal and 50 misbehavior), to handle large datasets efficiently.

**Hyperparameters:**
- Batch size: 64
- Number of epochs per subset: 10
- Total number of subsets: 10

The training process ensures that the model is exposed to the entire dataset without overwhelming system resources.


### Results

The trained autoencoder's performance is measured by the reconstruction error. The reconstruction loss is monitored to ensure the model is learning effectively.
- Number of misbehavior images in the top 100 high MSE images: **54**

<img width="920" alt="Screenshot 2024-08-06 at 11 49 32 AM" src="https://github.com/user-attachments/assets/e4756ad3-f459-44c0-9325-2a4a04c60c63">

Reconstructed images:
<img width="1070" alt="Screenshot 2024-08-06 at 11 49 07 AM" src="https://github.com/user-attachments/assets/22bcc0b3-4768-4a04-91c8-d092e531f96d">


### Model Checkpoints

Models are saved at each training step to ensure the best model can be retrieved and used for further analysis.
All the saved models are included in the repository.


