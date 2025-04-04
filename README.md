🩺 Chest X-Ray Classification using ANN (Artificial Neural Network)
This project aims to classify chest X-ray images into three categories: Covid, Normal, and Viral Pneumonia using a simple Artificial Neural Network (ANN) built with PyTorch.

📂 Dataset
The dataset is split into:

train/ : Contains training images for all three classes.

test/ : Contains images for evaluation.

Each class is stored in its own sub-directory:

css
Copy
Edit
Xray dataset/
├── train/
│   ├── Covid/
│   ├── Normal/
│   └── Viral Pneumonia/
└── test/
    ├── Covid/
    ├── Normal/
    └── Viral Pneumonia/
🧠 Model Architecture
A basic Feedforward Neural Network (ANN) is used:

Input Layer: Flattens image of size (120x120x3)

Hidden Layer: 64 neurons, ReLU activation

Output Layer: 3 classes (Covid, Normal, Viral Pneumonia)

📦 Dependencies
Install the required packages using:

bash
Copy
Edit
pip install torch torchvision matplotlib
Note: The model runs on Google Colab and mounts Google Drive to access the dataset.

🧪 Data Preprocessing
Images are transformed with:

Resizing to 120x120

Random horizontal flip for data augmentation

Normalization with pre-defined mean & std values

🧪 Training
Loss Function: CrossEntropyLoss

Optimizer: SGD with learning rate 0.001

Epochs: 10

Training/Validation split used from test set for performance monitoring

📈 Results
After training for 10 epochs, the model reached:

Training Accuracy: ~95%

Validation Accuracy: ~90%

Performance curves:

Accuracy vs Epochs

Loss vs Epochs

🧠 Inference
To make predictions on new images:

python
Copy
Edit
def predict_img(img, model):
    x = img.unsqueeze(0)
    y = model(x)
    pred = torch.argmax(y, dim=1)
    return train_data.classes[pred]
📸 Example Output
bash
Copy
Edit
Actual Label: Normal 
Predicted Label: Normal
The model correctly predicted the class of a test image.

📁 Directory Structure (in Google Colab)
python
Copy
Edit
drive.mount("/content/drive")

data_path_train = "/content/drive/MyDrive/Xray dataset/train"
data_path_test  = "/content/drive/MyDrive/Xray dataset/test"
📊 Visualizations
Helper function to visualize batches of images from the dataloader using matplotlib and torchvision.utils.make_grid.

✅ Future Improvements
Use a CNN instead of ANN for better feature extraction

Apply Transfer Learning (e.g., ResNet, EfficientNet)

Add more data augmentations

Hyperparameter tuning

🙌 Acknowledgement
Built by Swetanshu using PyTorch and Google Colab 🚀
