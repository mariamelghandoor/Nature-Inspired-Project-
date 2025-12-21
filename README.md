# Nature-Inspired Computing for Chest X-Ray Classification

This project leverages Nature-Inspired Computing (NIC) algorithms to optimize Deep Learning models for the classification of Chest X-Ray images. The system classifies images into three categories: **COVID-19**, **Pneumonia**, and **Normal**. Furthermore, it integrates Explainable AI (XAI) techniques to interpret and visualize the decision-making process of the models.

## 🚀 Project Overview

The primary goal of this repository is to demonstrate how bio-inspired optimization techniques can enhance the performance of Convolutional Neural Networks (CNNs). By automating the hyperparameter tuning process with algorithms like Particle Swarm Optimization (PSO) and Whale Optimization Algorithm (WOA), the project aims to achieve higher accuracy and robustness in medical image diagnosis.

## ✨ Key Features

*   **Deep Learning Architectures**: Implementation of custom CNNs and transfer learning using state-of-the-art models (e.g., VGG19, DenseNet201).
*   **Nature-Inspired Optimization**:
    *   **Particle Swarm Optimization (PSO)**: Used for optimizing model hyperparameters such as learning rate, batch size, and network structure.
    *   **Whale Optimization Algorithm (WOA)**: Employed to optimize the control parameters of PSO (e.g., C1 and C2 coefficients) or as a standalone optimizer.
    *   **Tabu Search**: Integrated for exploring solution spaces efficiently.
*   **Explainable AI (XAI)**:
    *   **SHAP (SHapley Additive exPlanations)**: For understanding feature importance.
    *   **LIME (Local Interpretable Model-agnostic Explanations)**: For local interpretability of predictions.
    *   **ELI5**: For debugging and visualizing machine learning classifiers.
*   **Performance Metrics**: Comprehensive evaluation using Confusion Matrix, ROC Curves, Precision, Recall, and Accuracy.

## 📂 Project Structure

The project consists of several Jupyter Notebooks, each focusing on specific aspects of the pipeline:

| Notebook | Description |
| :--- | :--- |
| `project_phase1.ipynb` | The foundational notebook covering data loading, preprocessing, distribution analysis, and baseline CNN model creation. |
| `optimizers_with_XAI.ipynb` | Features the core implementation of the hierarchy optimization strategy: **Whale Optimization → PSO → CNN**. It also includes XAI visualizations. |
| `lime-optmization-2-models.ipynb` | A comparative study focusing on optimizing different models and analyzing their decisions using LIME. |
| `Tabu_with Whale.ipynb` | Explores a hybrid approach involving Tabu Search and the Whale Optimization Algorithm. |

## 🛠️ Dependencies

Ensure you have the following Python libraries installed to run the notebooks:

```txt
tensorflow
keras
opencv-python
scikit-learn
matplotlib
seaborn
shap
eli5
lime
scikit-image
numpy
pandas
```

## 📊 Dataset

The models are trained on the **Chest X-Ray (Covid-19 & Pneumonia)** dataset. Ensure the dataset is structured correctly (e.g., `train/`, `test/` directories with subfolders for each class) before running the notebooks. By default, the code looks for data in `/kaggle/input/chest-xray-covid19-pneumonia` but can be adjusted to local paths.

## 🚀 Getting Started

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    ```
2.  **Install dependencies**:
    ```bash
    pip install tensorflow opencv-python scikit-learn matplotlib seaborn shap eli5 lime scikit-image
    ```
3.  **Run the notebooks**:
    Open the notebooks in Jupyter Lab or VS Code and run the cells sequentially. Start with `project_phase1.ipynb` for data setup and `optimizers_with_XAI.ipynb` for the main optimization pipeline.

## 📈 Results

The optimized models have demonstrated high accuracy (reaching ~97% in testing) in distinguishing between normal, viral pneumonia, and COVID-19 cases. The XAI visualizations provide heatmaps and boundary markers to highlight the lung regions contributing most to the diagnosis.
