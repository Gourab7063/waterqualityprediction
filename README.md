# 💧 Water Quality Prediction - RMS

This project aims to predict multiple water quality parameters using machine learning techniques—specifically, a `MultiOutputRegressor` wrapped around a `RandomForestRegressor`. It was developed as part of a one-month AICTE Virtual Internship sponsored by **Shell** in June 2025.

---

## 🌍 Overview

Access to clean water is a critical global concern. Accurate prediction of water quality metrics enables early pollution detection and timely intervention.

In this project, we:
- Collected and preprocessed real-world water quality datasets
- Performed supervised **multi-target regression**
- Built a modeling pipeline using **MultiOutputRegressor** with **RandomForestRegressor**
- Evaluated the model with appropriate regression metrics

---

## 🛠 Technologies Used

- **Python 3.12**
- `pandas`, `numpy` – Data handling
- `scikit-learn` – Machine learning and model evaluation
- `matplotlib`, `seaborn` – Visualizations
- **Jupyter Notebook** – Interactive experimentation

---

## 🔬 Predicted Parameters

The model predicts the following water quality indicators:
- NH₄ (Ammonium)
- BOD₅ (Biochemical Oxygen Demand)
- Colloids
- O₂ (Dissolved Oxygen)
- NO₃, NO₂ (Nitrates/Nitrites)
- SO₄ (Sulfates)
- PO₄ (Phosphates)
- Cl (Chloride)

---

## 📊 Model Performance

Evaluated using:
- **R² Score**
- **Mean Squared Error (MSE)**

Performance was consistent and acceptable across all predicted parameters.

---

## 📈 Sample Code Snippet

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 📎 Example Prediction

**Input:**
```json
{"pH": 7.2, "Temperature": 28.5, "Conductivity": 450}
```

**Output:**
```json
{"NH4": 0.3, "BOD5": 4.2, "Cl": 32.1, "...": "..."}
```

---

## 🔮 Future Work

- Test other ensemble regressors (e.g., XGBoost, LightGBM)
- Integrate time-series patterns if temporal data is available
- Deploy via Flask API or Streamlit for interactive predictions
- Visualize feature importances and prediction uncertainty

---

## 📁 Folder Structure

```
├── data/
├── notebooks/
├── models/
├── visualizations/
└── README.md
```

---

## 🤝 Acknowledgements

Special thanks to **AICTE** and **Shell** for organizing the virtual internship and providing the dataset and mentorship.

---

## 🧠 Author

- [Gourab7063](https://github.com/Gourab7063)

```
