# Sports Image Classification Project

## Overview
This project implements a deep learning model for classifying sports images into categories such as **Badminton, Cricket, Wrestling, Tennis, Soccer, Swimming, and Karate**. It uses a pre-trained **ResNet18** model from PyTorch's `torchvision` library, fine-tuned with transfer learning to classify images from a custom sports dataset. The project includes data preprocessing, model training, evaluation, and prediction, with **TensorBoard** logging for visualizing training progress.

---

## Dataset
The dataset is sourced from a Kaggle competition and consists of:

- **Training Set**: Contains 8,227 images with labels stored in `train.csv`. Each row includes an `image_ID` (e.g., `7c225f7b61.jpg`) and a `label` (e.g., Badminton, Cricket).
- **Test Set**: Contains images listed in `test.csv` without labels, used for generating predictions.

### Directory Structure
mohamedmostafam0-sports-image-classification/
├── sport-image-classification-project-deeplearning.ipynb  # Main Jupyter notebook
├── runs/                                                  # TensorBoard logs
├── checkpoint_L2_<weight_decay>.pth                       # Model checkpoints
├── best_model_L2_<weight_decay>.pth                       # Saved model weights
└── test_predictions.csv                                   # Test set predictions



### Class Distribution in Training Set:
- Cricket: 1,556 images  
- Wrestling: 1,471 images  
- Tennis: 1,445 images  
- Badminton: 1,394 images  
- Soccer: 1,188 images  
- Swimming: 595 images  
- Karate: 578 images  

---

## Requirements
To run this project, you need the following dependencies:

- Python 3.8+
- PyTorch
- Torchvision
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TQDM
- TensorBoard
- Pillow (PIL)
- Pyngrok *(optional, for TensorBoard tunneling)*

Install dependencies using:

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn tqdm tensorboard pillow pyngrok
```

## Setup
1. Clone the Repository:
```bash
git clone https://github.com/mohamedmostafam0/Sports-Image-Classification.git
cd mohamedmostafam0-sports-image-classification
```
2. Prepare the Dataset:
- Download the dataset from Kaggle: Sports Image Classification Dataset.
- Place it in the appropriate directory (e.g., /kaggle/input/sports-image-classification/ if running on Kaggle).
  
3. Set Up TensorBoard:
```bash
tensorboard --logdir runs
```
4. Environment:
- Developed on Kaggle with GPU support.
- Ensure a CUDA-compatible GPU is available for faster training.

  ## Usage

The notebook `sport-image-classification-project-deeplearning.ipynb` contains the complete workflow.

### Key Steps:

1. **Data Loading and Exploration**:
   - Loads `train.csv` and `test.csv` using Pandas.
   - Visualizes class distribution using bar plots.

2. **Data Preprocessing**:
   - Cleans data (null values, duplicates).
   - Encodes labels with `LabelEncoder`.
   - Applies image transformations via `torchvision.transforms`.

3. **Model Training**:
   - Uses pre-trained ResNet18 with modified FC layer (Dropout=0.3).
   - Trained for 5 epochs using Adam optimizer (LR=0.001), CrossEntropyLoss.
   - Tests L2 regularization values: 0.0, 1e-4, 1e-3.
   - Saves best model weights and checkpoints.

4. **Evaluation**:
   - Evaluates training set after each run.
   - Logs metrics to TensorBoard.

5. **Prediction**:
   - Generates predictions for test set (`test_predictions.csv`).
   - Includes a function for single-image prediction.

#Run the Notebook:
```bash
jupyter notebook sport-image-classification-project-deeplearning.ipynb
```

## Example Prediction

```python
image_path = "/kaggle/input/sports-image-classification/dataset/test/91b5f8b7a9.jpg"
predicted_class = predict_image(model, image_path, train_transform, device, classes)
print(f"Predicted Class: {predicted_class}")
```

Example Output:
```bash
Predicted Class: Badminton
```

## Results

Training accuracy improved over 5 epochs. For weight decay = 0.0:

- **Epoch 1:** Accuracy = 68.06%
- **Epoch 2:** Accuracy = 87.03%, Loss = 51.8579
- **Epoch 3:** Accuracy = 93.31%, Loss = 26.0042
- **Epoch 4:** Accuracy = 95.15%, Loss = 19.1418
- **Epoch 5:** Accuracy = 96.92%, Loss = 11.7459

**Final training accuracy:** **96.92%**

Results for other weight decay values are available in **TensorBoard** logs.

---

## Notes

- This project assumes execution in a **Kaggle** environment. Modify file paths if running locally.
- The `google.colab.drive.mount` call is not needed and can be removed for Kaggle.
- TensorBoard logs are stored in the `runs/` directory.
- Final predictions are saved in `test_predictions.csv`.

