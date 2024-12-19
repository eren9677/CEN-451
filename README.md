# Mushroom Edibility Prediction Project

This presentation outlines a project focused on using a **Multi-Layer Perceptron (MLP) neural network** to predict whether mushrooms are edible or poisonous. The project utilises the **UCI Mushroom Dataset** and aims to provide an overview of the development process, the methodologies employed, and the results achieved.

## Project Overview
*   **Goal:** To predict mushroom edibility using a 3-layered MLP.
*   **Dataset:** UCI Mushroom Dataset.
*   **Skills Developed**: ML project development, programming (PyTorch, MLOps), teamwork, and task division.
*   **Technologies Used**:
    *   **Python**: Core programming language.
    *   **PyTorch**: Building ML blocks.
    *   **Matplotlib & Seaborn**: Plotting and monitoring.
    *   **Scikit-learn**: Applying metrics.
    *   **NumPy**: Array operations.
    *  **Pandas**: Statistical data analysis.
    *  **Git & GitHub**: Version control and collaboration.
    *   **VS Code**: Development environment.

## Dataset
*   **Source:** UCI Mushroom Dataset.
*   **Creator:** Jeff Schlimmer .
*   **Size:** 8,124 samples .
*   **Species:** 23 species of gilled mushrooms .
*   **Families:** Agaricus and Lepiota.
*   **Classes:** Edible or Poisonous.
*   **Attributes:** 23 categorical attributes.
*   **Characteristics:** No missing values, balanced distribution of classes.
*   **Key Attributes**:  class, cap shape, colour, surface, odour, gill attachment, spacing, size and colour, and stalk features.

## Methodology

### Data Preprocessing & EDA
*   **Exploratory Data Analysis (EDA):** Used to understand data patterns, distributions, correlations, and check for missing values.
*   **Findings:**
    *   "Veil-Type" had only one unique value and was dropped.
    *   "Odor" was identified as a strong predictor of edibility.
*   **Preprocessing Steps**:
    *   One-hot encoding of categorical features 
    *   Data split into training (70%), validation (15%), and test (15%) sets .

### Model Architecture
*   **Type**: 3-layer MLP.
*   **Input Layer:** Receives one-hot encoded features.
*   **Hidden Layer:** 10 neurons.
*   **Output Layer:** Single neuron with sigmoid activation.
*   **Reason for choosing MLP:** Suitable for structured data, efficient, and easier to implement than more complex architectures [1, 11].

### Model Implementation
*   **Linear Transformations**:  Implemented using `nn.Linear`.
*    **Batch Normalization**: Used to prevent internal covariate shift with `nn.BatchNorm1d`.
*    **Activation Functions**:
    *   **ReLU**: Used in hidden layers to prevent vanishing gradients.
    *   **Sigmoid**: Used in the output layer for binary classification, scales output to probability.
*   **Forward pass**: Implements the flow of data through the network.

### Training and Validation
*   **Training Process**:
    *   Data fed in batches of 64.
    *   Forward pass, batch normalisation and ReLU activation is applied between layers .
    *   **Binary Cross-Entropy loss** calculated.
    *   **Adam optimizer** used to update model parameters .
    *   Trained for 15 epochs.
*    **Validation**: Used to monitor performance on unseen data and prevent overfitting .

### Model Testing
*   Evaluated on an unseen test dataset.
*   **Test Accuracy**: 99.75% .
*   **Test Loss**: 0.0001 .
*  **Classification Report**: Provides precision, recall, and F1 scores.
*   **Confusion Matrix**: Visualised to show the model's performance in classifying edible and poisonous mushrooms .

## Results
*  **High test accuracy** .
*   **Strong classification performance** as shown in the confusion matrix .
*   The model successfully learned to differentiate between edible and poisonous mushrooms .

## Observations and Challenges
*   **Errors Encountered**:
    *   Incorrect handling of categorical features.
    *   Batch size selection.
    *   Number of neurons in hidden layers.
    *   Activation function choice 
*   **Learning Points:**
    *   Importance of careful data preprocessing.
    *   Tuning hyperparameters for optimal performance.
    *   Selecting appropriate activation functions (ReLU performed better than Tanh) .

## Conclusion
*   The project successfully designed, implemented, and optimized an MLP for mushroom edibility classification .
*   Gained skills in Python, PyTorch, data analysis, and visualization .
*  The project serves as a milestone for future work in applying neural networks to realworld datasets .

