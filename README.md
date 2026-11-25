Epileptic Seizure Recognition Based on Multi-domain EEG Features and Machine Learning

1. Project Introduction

This project implements an EEG-based epileptic seizure recognition system using multi-domain feature extraction and machine learning algorithms.

Support Vector Machine (SVM) and Random Forest are employed to compare the classification performance of different feature combinations.

2. Main Functions

- EEG data loading and preprocessing
- Time-domain feature extraction
- Frequency-domain feature extraction
- Wavelet feature extraction
- Multi-domain feature fusion
- SVM classification
- Random Forest classification
- Feature combination comparison
- Group-based final model evaluatio

3. Experimental Environment

- Windows
- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- PyWavelets
- Scikit-learn

4. Installation

Install the required Python packages:

`pip install -r requirements.txt`

5. Usage

Run the Python scripts sequentially from Step 01 to Step 09.

To perform the final evaluation after completing the preceding steps:

`python 09_final_evaluation.py`

Experimental figures and evaluation results are stored in the figures and results directories.

6. Experimental Results

| Feature Set | Model         | Accuracy | F1-score | AUC    |
| ----------- | ------------- | -------- | -------- | ------ |
| 17 Features | SVM           | 97.96%   | 0.9513   | 0.9932 |
| 17 Features | Random Forest | 98.52%   | 0.9642   | 0.9993 |
| 27 Features | SVM           | 98.00%   | 0.9521   | 0.9960 |
| 27 Features | Random Forest | 99.22%   | 0.9807   | 0.9995 |

The selected frequency-domain and wavelet features reduce the feature dimension from 27 to 17, achieving a dimensionality reduction of 37.04%.

The Random Forest classifier using all 27 features achieves the highest overall accuracy on the final test set.


Integrated Circuit Design and Integrated Systems

7. Disclaimer

This project is developed for academic and educational purposes and is not intended for clinical diagnosis.
