
# Enhancing Diabetes Complications Prediction Through Knowledge Graphs and Convolutional Networks
## Overview

**KGDCP** is a deep learning model designed to predict diabetes complications by leveraging knowledge graphs and advanced neural networks. The model integrates a physical examination knowledge graph with an improved Diabetes Knowledge Graph (DiaKG) using self-attention-based Convolutional Neural Networks. This approach helps in early prediction of diabetes complications, such as cerebral infarction and peripheral neuropathy, offering a more accurate and comprehensive understanding of the disease and its progression.


## Features

- **Integration of Knowledge Graphs**: The model combines a **Physical Examination Knowledge Graph** and an improved **DiaKG**, enhancing the prediction accuracy.
- **Self-Attention Mechanism**: Uses a token-to-token self-attention to capture dependencies between examination indicators and source-to-token attention for diagnostic entities.
- **Convolutional Neural Networks (CNN)**: Extracts local features to generate representation vectors that improve prediction performance.
- **Improved Knowledge Graphs**: Knowledge deduplication and coreference resolution enhance the comprehensiveness and accuracy of the DiaKG.
- **Prediction of Complications**: The model outperforms traditional models in predicting cerebral infarction and peripheral neuropathy in diabetes patients.

## Dataset

The dataset used for training and evaluation consists of **diabetes patients** aged between **27 to 90 years**. Both **male** and **female** patients are included in the dataset. The dataset provides comprehensive physical examination data, diagnostic information, and the complications each patient faces, specifically focusing on cerebral infarction and peripheral neuropathy.

### Data Details
- **Patient Age Range**: 27–90 years
- **Gender**: Both male and female patients
- The dataset used in our experiments comprises 1312 records, each corresponding to a unique patient, collected over a 5-year period (2018–2022). Each record includes 18 features, covering physical examination data, diagnostic information, complications, and demographic attributes. The dataset is divided into a 4:1 ratio for training and testing, ensuring a robust evaluation of our model. The age distribution of patients ranges from 27 to 90 years, with both male and female patients represented, providing a diverse and clinically relevant sample for diabetes complications prediction. To ensure data quality, we include only patients with complete records and excluded those with missing or inconsistent data. Outliers are identified and corrected based on clinical guidelines. Due to privacy concerns, only sample data from the provided dataset is shared here, which includes physical examination data, diagnostic information, and the complications associated with each patient.

Note: Due to privacy concerns, only sample data is provided in this repository.

## Model Architecture

### Knowledge Graph Integration
- **Physical Examination Knowledge Graph**: Based on normal reference ranges of examination indicators, this graph helps to enhance the interpretation of the physical examination data.
- **Improved DiaKG**: This knowledge graph integrates diabetes-specific information and is enhanced through knowledge deduplication and coreference resolution for improved accuracy.

### Neural Network Components
- **Token-to-Token Self-Attention**: Captures dependencies among various physical examination indicators.
- **Convolutional Neural Networks (CNN)**: Extracts local features from the knowledge graph embeddings and patient data to generate meaningful representations.
- **Source-to-Token Attention**: Assesses dependencies between diagnostic entities and their relationships with the entire entity set.
- **Entity-to-Physical Examination Attention**: Gauges the relevance of diagnostic entities to physical examination data.

These components together allow the model to generate accurate predictions for complications related to diabetes, such as **cerebral infarction** and **peripheral neuropathy**.

## Installation

To run the KGDCP model locally, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/KGDCP.git
cd KGDCP
pip install -r requirements.txt
