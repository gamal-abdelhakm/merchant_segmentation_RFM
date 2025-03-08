# Customer Recommendation System

This repository implements a recommendation system based on Customer Segmentation by RFM Clustering.

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://share.streamlit.io/gamal-abdelhakm/merchant_segmentation_RFM/main/app.py)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Data Sources](#data-sources)
- [Technologies Used](#technologies-used)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/gamal-abdelhakm/merchant_segmentation_RFM.git
    ```
2. Navigate to the project directory:
    ```bash
    cd merchant_segmentation_RFM
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On MacOS/Linux:
        ```bash
        source venv/bin/activate
        ```
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to:
    ```
    http://localhost:8501
    ```

## Features

### User Profile & Clustering
- Enter your demographic information to find out which user cluster you belong to.
- View cluster demographics and top recommended categories and merchants.

### Recommendation Modes
- **User ID**: Get personalized recommendations based on user ID.
- **Custom Features**: Receive recommendations by adjusting custom user profile features.

### Visualizations
- Customer segment visualization using PCA.
- Category preferences and top recommended merchants.

### Detailed Recommendations
- Detailed list of top categories and merchants for each category.

## Data Sources

This system uses several datasets:
- **Merchant Transaction Data**: Includes transaction ranks, points, values, ages, and categories.

## Technologies Used

- **Streamlit**: Web interface framework.
- **Scikit-learn**: Machine learning algorithms for clustering.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualizations.
- **Plotly**: Interactive visualizations.
