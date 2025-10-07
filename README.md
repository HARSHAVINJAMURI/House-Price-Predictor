# House Price Prediction App

This is a **House Price Prediction Application** built using Python, Streamlit, and Scikit-Learn. The app allows users to input key property features and predicts the expected price in lakhs.

## Features

* User-friendly **Streamlit interface** for inputs.
* Predicts **house prices** based on:

  * Number of Bathrooms
  * Number of Balconies
  * Size (in terms of bedrooms)
  * Total Square Feet
* Uses **Random Forest Regressor** for prediction.
* Handles missing data, duplicates, and feature scaling.

## Tech Stack

* Python 3.10
* Pandas
* Numpy
* Scikit-Learn
* Streamlit

## Installation

1. Clone this repository:

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Input the property details in the sidebar or input fields:

   * Number of Bathrooms
   * Number of Balconies
   * Size (bedrooms)
   * Total Square Feet

3. Click **Predict Price** to see the predicted house price.

## Dataset

The app uses the **Bengaluru House Price Dataset**, which includes features such as:

* `bath`, `balcony`, `size`, `total_sqft`, `price`

Ensure that your CSV file is properly formatted and matches these columns if using a custom dataset.

## Preprocessing

* Missing values are filled with median values for numeric columns.
* `size` is converted from strings to integers.
* `total_sqft` is converted from string formats to floats.
* Unnecessary columns (`society`, `area_type`, `availability`, `location`) are dropped.
* Duplicates are removed.
* Features are scaled using `MinMaxScaler`.

## Model

* Random Forest Regressor is trained on the preprocessed dataset.
* Predictions are made using scaled feature values.
* Metrics like **R² Score** and **Mean Squared Error** can be computed if needed.

## File Structure

```
├── app.py               # Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── Bengaluru_House_Data.csv  # Dataset
```

## Dependencies

Listed in `requirements.txt`.

---

## 🧑‍💻 Author

**Developed by:** Harsha Vinjamuri (Data Science & ML Enthusiast)
---

## 📄 License

This project is released under the **MIT License** — free to use and modify for educational or commercial purposes.

---
