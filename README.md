# Traffic Sign Recognition 

This project implements a traffic sign recognition system using Convolutional Neural Networks (CNNs) in Python with Keras and TensorFlow. It classifies traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset, preprocesses images, trains models, evaluates performance, and compares a custom CNN with a pre-trained MobileNet model. Data augmentation is applied to improve model robustness.

## Project Description

- **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark) from Kaggle. It contains images of 43 different traffic sign classes.
- **Preprocessing**: Resize images to 32x32 pixels and normalize pixel values (0-1 range).
- **Model Training**: 
  - Build and train a custom CNN for multi-class classification.
  - Use data augmentation (rotation, zoom, shifts) to enhance training data.
- **Evaluation**: Measure performance using accuracy and confusion matrix.
- **Comparison**: Compare the custom CNN with a pre-trained MobileNet model (transfer learning).
- **Topics Covered**: Computer Vision (CNN), Multi-class Classification.
- **Bonus Features**: Data augmentation for better generalization; comparison between custom and pre-trained models.

## Dataset

The dataset used is the [GTSRB - German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) from Kaggle. It includes thousands of labeled traffic sign images divided into training and testing sets.

### How to Download and Prepare the Dataset
1. Download the dataset from the Kaggle link above.
2. Extract the files.
3. Place the training images in a directory structured as `dataset/Train/<class_id>/<image_files>`, where `<class_id>` is the folder for each traffic sign class (0-42).
4. The code assumes this structure for loading data. If your path differs, update `data_dir = "dataset\\Train"` in the notebook.

## Requirements

- Python 3.10+
- Libraries:
  - numpy
  - opencv-python (cv2)
  - tensorflow (including keras)
  - scikit-learn
  - matplotlib
  - tabulate

Install dependencies using:
```
pip install numpy opencv-python tensorflow scikit-learn matplotlib tabulate
```

## How to Run

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```

2. Download and prepare the dataset as described above.

3. Open the Jupyter notebook:
   ```
   jupyter notebook code.ipynb
   ```

4. Run the cells sequentially:
   - Load and preprocess data.
   - Split into train/test sets.
   - Apply data augmentation.
   - Train the custom CNN.
   - Train the MobileNet model.
   - Evaluate both models (accuracy and confusion matrices).
   - View plots and tabular comparison.

Note: Training may take time depending on your hardware. The code uses 10 epochs by default; adjust as needed.

## Code Structure

- **code.ipynb**: The main Jupyter notebook containing all steps:
  - Data loading and preprocessing.
  - Data augmentation with `ImageDataGenerator`.
  - Custom CNN model definition and training.
  - MobileNet model (pre-trained) definition and training.
  - Evaluation: Accuracy, confusion matrices, and plots.
  - Tabular comparison of model accuracies.

Key snippets:
- Custom CNN architecture:
  ```python
  custom_cnn = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
      MaxPooling2D(pool_size=(2, 2)),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D(pool_size=(2, 2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(len(np.unique(y)), activation='softmax')
  ])
  ```
- MobileNet (transfer learning):
  ```python
  base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
  mobilenet_model = Sequential([
      base_model,
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(len(np.unique(y)), activation='softmax')
  ])
  ```

## Results

- **Custom CNN**:
  - Accuracy: ~98.56% (on test set after 10 epochs).
  - Training history: Accuracy improves from ~32% to ~87% over epochs.
  - Confusion Matrix: Plotted to show misclassifications.

- **MobileNet**:
  - Accuracy: ~97.25% (slightly lower than custom CNN in this setup, but faster convergence).
  - Confusion Matrix: Plotted for comparison.

- **Comparison Table** (example output):
  ```
  +------------+------------+
  | Model      |   Accuracy |
  +============+============+
  | Custom CNN |   0.98559  |
  +------------+------------+
  | MobileNet  |   0.972456 |
  +------------+------------+
  ```

Plots of confusion matrices are generated in the notebook for visual analysis.

## Potential Improvements

- Increase epochs or fine-tune hyperparameters for better accuracy.
- Add more augmentation techniques (e.g., brightness adjustments).
- Test on real-world images or the official GTSRB test set.
- Deploy the model (e.g., via TensorFlow Lite for mobile).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: Provided by Kaggle user [meowmeowmeowmeowmeow](https://www.kaggle.com/meowmeowmeowmeowmeow).
- Libraries: Thanks to the TensorFlow/Keras and OpenCV communities.
