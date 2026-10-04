# E-Commerce Customer Sales Analytics-ML

An end-to-end **Machine Learning extension for an E-Commerce Customer Sales project**, focused on understanding customer behavior through **customer segmentation, purchase prediction, and customer value analysis**.

This project extends an existing e-commerce dataset by applying multiple machine learning techniques to identify customer groups, predict repeat purchasing behavior, and estimate future customer spending.

The project contains **three separate Machine Learning components**:

1. **Customer Segmentation**
2. **Purchase Prediction**
3. **Customer Value Analysis**

The original **Data Analysis & Business Intelligence / Power BI work is maintained separately** and is not included as a notebook workflow in this repository.

---

## Project Objectives

* Build customer-level behavioral features from transaction data
* Segment customers using multiple clustering algorithms
* Compare clustering techniques using Silhouette Score
* Reduce customer behavioral dimensions using PCA
* Predict whether a customer will purchase again within 30 days
* Engineer historical and recent customer purchasing behavior features
* Compare multiple classification models
* Analyze classification performance using several evaluation metrics
* Estimate future 90-day customer spending
* Compare regression models for customer value prediction
* Generate customer-level predictions and business-oriented customer segments
* Save trained models and ML results for further use

---

# Machine Learning Components

## 1. Customer Segmentation

The first ML component groups customers according to their historical purchasing behavior.

### Customer-Level Features

The transaction-level dataset is aggregated into customer-level features including:

* Total Sales
* Total Orders
* Total Quantity
* Average Order Value
* Recency

Recency represents the number of days since the customer's last purchase.

### Feature Scaling

Because the customer features have different numerical scales, `StandardScaler` is used before clustering.

### Clustering Algorithms

Three clustering algorithms are explored:

* **K-Means**
* **Agglomerative Clustering**
* **DBSCAN**

K-Means is used for the main customer segmentation workflow, while Agglomerative Clustering and DBSCAN are used for algorithm comparison.

### Cluster Selection

The K-Means clustering process includes:

* Elbow Method
* Inertia analysis
* Silhouette Score

The clustering algorithms are compared using Silhouette Score to evaluate cluster separation.

### PCA-Based Customer Segmentation

A second customer segmentation workflow uses additional customer behavioral features:

* Recency Days
* Order Count
* Total Spending
* Total Quantity
* Average Order Value
* Customer Lifetime Days
* Purchase Frequency

PCA is applied to transform the standardized customer features into principal components.

The first two principal components are then visualized to understand customer behavior in a lower-dimensional space.

K-Means clustering is subsequently applied to the standardized customer behavioral features.

### Customer Segments

The final segmentation contains two business-oriented customer groups:

* **Higher-Value Active Customers**
* **Lower-Value Less-Active Customers**

The segments are interpreted using:

* Historical spending
* Order frequency
* Quantity purchased
* Average order value
* Recency
* Customer lifetime
* Purchase frequency
* Future 90-day spending

### Business Applications

**Higher-Value Active Customers**

Possible strategies include:

* Customer retention
* Loyalty programs
* Personalized recommendations
* Cross-selling
* Premium offers

**Lower-Value Less-Active Customers**

Possible strategies include:

* Re-engagement campaigns
* Personalized promotions
* Discount offers
* Product recommendations
* Customer retention campaigns

---

# 2. Purchase Prediction

The second ML component predicts whether a customer will make another purchase **within 30 days**.

This is formulated as a **binary classification problem**.

### Target Variable

`Purchased_Within_30_Days`

The target is created using the customer's next observed purchase date.

* `1` → Customer purchases again within 30 days
* `0` → Customer's next purchase is more than 30 days later
* Missing → Future outcome is not observable

The dataset is filtered so that observations near the end of the dataset do not incorrectly receive an unknown future outcome.

---

## Temporal Feature Engineering

Historical customer behavior is used to create predictive features.

### Previous Purchase Features

* Days Since Previous Purchase
* Previous Order Count
* Previous Purchase Amount
* Previous Purchase Quantity
* Previous Total Spending
* Previous Average Order Value
* Customer Lifetime Days

### Recent Purchase Activity

* Orders Last 7 Days
* Orders Last 30 Days
* Orders Last 60 Days
* Orders Last 90 Days

### Recent Spending

* Spending Last 30 Days
* Spending Last 60 Days
* Spending Last 90 Days

### Recent Quantity

* Quantity Last 30 Days
* Quantity Last 60 Days
* Quantity Last 90 Days

### Purchase Frequency

* Average Days Between Purchases
* Median Days Between Purchases
* Purchase Frequency

Historical features are calculated using previous transactions to reduce the risk of future information leaking into the prediction features.

---

## Time-Based Train/Test Split

Instead of randomly splitting the transactions, the enhanced prediction dataset is divided according to transaction time.

Approximately:

* Earlier observations → Training data
* Later observations → Testing data

This better represents a real-world prediction scenario where a model is trained using historical customer behavior and then applied to future customers.

Missing values in the temporal behavioral features are imputed using **medians calculated from the training dataset**.

Feature scaling is performed using `StandardScaler`.

---

## Classification Models

Four classification algorithms are compared:

### Logistic Regression

Used as a strong and interpretable classification baseline.

### Support Vector Machine

An RBF-kernel SVM is used to model potentially nonlinear relationships between customer behavior and repeat purchasing.

### K-Nearest Neighbors

KNN predicts customer behavior based on similarities between historical customer behavior patterns.

### MLP Neural Network

A Multi-Layer Perceptron is used to model more complex nonlinear relationships.

The MLP architecture includes hidden layers and uses:

* ReLU activation
* Adam optimizer
* Early stopping

An additional MLP configuration is also experimented with to investigate a deeper neural-network architecture.

---

## Classification Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Classification Report

Additional evaluation includes:

* ROC Curves
* Precision-Recall Curves
* Probability distribution analysis
* Classification threshold analysis
* MLP training-loss analysis

Threshold analysis is performed to investigate the trade-off between precision and recall at different prediction probability thresholds.

---

# 3. Customer Value Analysis

The third ML component estimates a customer's **future 90-day spending** using historical customer behavior.

This is formulated as a **regression problem**.

---

## Observation and Future Period

A fixed observation date is used:

**30 June 2025**

Historical customer behavior is calculated using transactions up to this date.

The target is the customer's spending during the following 90-day period.

### Target

`Future_90_Day_Spend`

Customers without future purchases are assigned a future spending value of zero.

---

## Customer Value Features

The regression models use:

* Recency Days
* Order Count
* Total Spending
* Total Quantity
* Average Order Value
* Customer Lifetime Days
* Purchase Frequency

The existing customer-level historical features are therefore used to estimate future customer spending.

---

## Regression Models

Three regression algorithms are compared:

### Linear Regression

Provides a simple baseline for predicting future customer spending.

### Random Forest Regressor

Captures nonlinear relationships between historical customer behavior and future spending.

### XGBoost Regressor

Uses gradient boosting to model complex relationships between customer features and future spending.

---

## Regression Evaluation

The models are evaluated using:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — Coefficient of Determination

The models are compared using a consolidated regression results table.

The final predictions are used to generate a customer-level:

`Predicted_90_Day_Value`

Negative predicted values are clipped to zero.

This allows customers to be ranked according to their estimated future spending value.

---

# PCA and Customer Behavior Analysis

PCA is also used as part of the customer intelligence workflow.

The customer behavioral features are standardized before PCA.

The PCA workflow includes:

1. Feature preparation
2. Missing-value handling
3. Standardization
4. Principal Component Analysis
5. Explained variance analysis
6. PCA loading analysis
7. Two-dimensional customer visualization
8. K-Means segmentation in the behavioral feature space

This provides a lower-dimensional representation of customer behavior while retaining important variation in the original features.

---

# Project Structure

```text
E-Commerce-Customer-Sales-Analysis/
│
├── data/
│   ├── raw/
│   │   ├── customer_master.csv
│   │   ├── dataset_statistics.csv
│   │   ├── ecommerce_sales_customer_analytics_150k.csv
│   │   ├── order_items.csv
│   │   └── product_catalog.csv
│   │
│   └── processed/
│       ├── category_sales.csv
│       ├── ecommerce_cleaned.csv
│       └── product_summary.csv
│
├── notebooks/
│   ├── 01_ml_model.ipynb
│   ├── 02_ml_model.ipynb
│   └── 03_ml_model.ipynb
│
├── results/
│   ├── cluster_results/
│   └── clustering_comparison.csv
│
├── models/
│   ├── customer_segmentation_scaler.pkl
│   └── customer_segmentation_kmeans.pkl
│
└── .gitignore
```

> **Note:** The `processed` datasets were created during the earlier Data Analysis & Business Intelligence / Power BI work. The Power BI/Data Analysis workflow is maintained separately from this ML repository.

---

# Notebook Organization

## `01_ml_model.ipynb` — Customer Segmentation

This notebook contains the customer segmentation workflow, including:

* Transaction data loading
* Customer-level feature engineering
* Recency calculation
* Feature scaling
* K-Means clustering
* Elbow Method
* Cluster profiling
* Agglomerative Clustering
* DBSCAN
* Silhouette Score comparison
* Customer segmentation analysis
* PCA-based customer behavior analysis
* Final customer segment creation
* Segment visualization
* Customer segmentation result export
* Model/scaler persistence

---

## `02_ml_model.ipynb` — Purchase Prediction

This notebook contains the customer repeat-purchase prediction workflow, including:

* Transaction data preparation
* Next purchase calculation
* 30-day purchase target creation
* Previous purchase feature engineering
* Historical customer behavior features
* Recent order activity features
* Recent spending features
* Recent quantity features
* Purchase frequency features
* Outcome observability filtering
* Time-based train/test splitting
* Training-set median imputation
* Feature scaling
* Logistic Regression
* SVM
* KNN
* MLP Neural Network
* Classification evaluation
* Confusion matrices
* ROC curves
* Precision-Recall curves
* Threshold analysis
* MLP loss analysis

---

## `03_ml_model.ipynb` — Customer Value Analysis

This notebook contains the future customer value prediction workflow, including:

* Historical/future time-window creation
* Customer-level feature engineering
* Future 90-day spending calculation
* Future purchase analysis
* Missing and infinite value handling
* Train/test split
* Linear Regression
* Random Forest Regression
* XGBoost Regression
* MAE evaluation
* RMSE evaluation
* R² evaluation
* Customer future-value prediction
* Customer ranking
* PCA
* PCA loadings
* Customer behavior visualization
* K-Means segmentation
* Cluster profiling
* Silhouette analysis
* Business-oriented customer segment labeling

---

# Dataset

The project uses an existing e-commerce transaction dataset containing customer, order, product, sales, and purchasing information.

### Raw Data

```text
data/raw/
├── customer_master.csv
├── dataset_statistics.csv
├── ecommerce_sales_customer_analytics_150k.csv
├── order_items.csv
└── product_catalog.csv
```

The main transaction dataset used by the ML notebooks is:

```text
ecommerce_sales_customer_analytics_150k.csv
```

The ML workflows primarily use transaction-level information such as:

* Customer ID
* Order ID
* Order Date
* Net Sales
* Quantity
* Customer purchasing history

---

# Processed Data

```text
data/processed/
├── category_sales.csv
├── ecommerce_cleaned.csv
└── product_summary.csv
```

These processed datasets were generated during the earlier **Data Analysis & Business Intelligence / Power BI project** and are retained in the existing project structure.

They are not the primary datasets created by the three ML notebooks.

---

# Machine Learning Workflow

```text
Raw E-Commerce Transactions
            │
            ▼
   Customer-Level Features
            │
      ┌─────┼───────────────┐
      │     │               │
      ▼     ▼               ▼
Segmentation  Purchase      Customer
              Prediction    Value Analysis
      │           │              │
      ▼           ▼              ▼
Clustering   Classification   Regression
      │           │              │
      ▼           ▼              ▼
Customer     Repeat Purchase   Future 90-Day
Segments     Prediction        Spending
      │           │              │
      └───────────┼──────────────┘
                  ▼
        Customer Intelligence
```

---

# Technologies and Libraries

### Programming

* Python
* Jupyter Notebook

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Machine Learning Techniques

* Feature Engineering
* Standardization
* K-Means Clustering
* Agglomerative Clustering
* DBSCAN
* PCA
* Logistic Regression
* Support Vector Machine
* K-Nearest Neighbors
* Multi-Layer Perceptron
* Linear Regression
* Random Forest Regression
* XGBoost Regression

### Model Evaluation

* Silhouette Score
* Inertia / Elbow Method
* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* MAE
* RMSE
* R²

---

# Key Machine Learning Concepts Demonstrated

This project demonstrates practical application of:

* Customer-level feature engineering
* Behavioral feature engineering
* Temporal feature engineering
* Customer segmentation
* Unsupervised learning
* Supervised classification
* Supervised regression
* Dimensionality reduction
* PCA
* Model comparison
* Time-based validation
* Data leakage prevention
* Missing-value handling
* Feature scaling
* Classification threshold analysis
* Model evaluation
* Customer value prediction
* Business-oriented ML interpretation

---

# Data Leakage Prevention

The purchase prediction workflow is designed to avoid using future information when constructing historical customer features.

For example, previous purchase information is calculated using shifted historical transactions rather than the customer's future purchase.

The prediction dataset also excludes observations where the 30-day outcome cannot be observed because they occur too close to the end of the dataset.

For the enhanced purchase prediction workflow, the train/test split is time-based rather than purely random.

Missing values for selected temporal features are imputed using statistics calculated only from the training dataset.

These steps make the prediction setup closer to a real-world forecasting scenario.

---

# Business Applications

The ML results can support e-commerce decision-making in several areas.

### Customer Segmentation

Identify different customer behavior groups and design targeted marketing strategies.

### Repeat Purchase Prediction

Identify customers who are more likely to purchase again within 30 days.

Potential applications include:

* Retention campaigns
* Personalized recommendations
* Promotional targeting
* Customer engagement

### Customer Value Prediction

Estimate future customer spending to help prioritize high-value customers.

Potential applications include:

* Customer prioritization
* Loyalty programs
* Marketing resource allocation
* Personalized offers
* Customer retention

---

# Relationship to the Original E-Commerce Project

This repository is an **ML extension of an existing E-Commerce Customer Sales project**.

The project is divided conceptually into:

```text
E-Commerce Project
│
├── Data Analysis & Business Intelligence
│   └── Maintained separately
│
└── Machine Learning Extension
    ├── Customer Segmentation
    ├── Purchase Prediction
    └── Customer Value Analysis
```

The Machine Learning component adds predictive and unsupervised learning techniques on top of the existing e-commerce transaction data.

---

# Future Improvements

Possible future improvements include:

* Hyperparameter optimization
* More advanced customer behavioral features
* Better handling of highly skewed customer spending
* Advanced temporal validation strategies
* Probability calibration
* Feature importance and model explainability
* Customer-level prediction dashboards
* Integration with Power BI
* Deployment of selected ML models through an API
* Automated customer scoring pipelines

---

# Author

**Hirushi Fernando**

BSc (Hons) Computer Science
AI / Machine Learning & Data Science Enthusiast

---

## Project Focus

**E-Commerce Customer Intelligence using Machine Learning**

> Customer Segmentation + Purchase Prediction + Customer Value Analysis
