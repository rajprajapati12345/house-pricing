# Mumbai House Price Prediction

Machine learning project jo Mumbai ke ghar ka price (lakh mein) predict karta hai, area, BHK, region, type, status aur age ke basis par.

## Problem
Mumbai mein ghar ka price kai cheezon par depend karta hai. Is project mein ek model banaya gaya jo in details se price ka andaza lagata hai.

## Dataset
- File: Mumbai_House_Prices.csv (Kaggle)
- Original size: 76,038 rows, 9 columns (bhk, type, locality, area, price, price_unit, region, status, age)
- Duplicates hatane ke baad: [ ] rows
- Outliers hatane ke baad final data: [ ] rows

## Steps
1. Data loading aur inspection (Pandas)
2. Cleaning: price ko lakh mein convert kiya (Cr aur L alag the), duplicates hataye
3. Outlier removal: price per sqft aur area per BHK ke basis par
4. Feature engineering: price_per_sqft, area_per_bhk, rare regions ko "Other" mein daala, locality hata di
5. EDA: price distribution, area vs price, top regions, BHK vs price, correlation heatmap
6. Encoding: one-hot encoding (type, region, status, age)
7. Train-test split (80/20)
8. Models: Linear Regression, Random Forest, Gradient Boosting
9. Hyperparameter tuning: RandomizedSearchCV (cross-validation)
10. Feature importance aur prediction function

## Results
| Model | MAE (lakh) | RMSE (lakh) | R2 |
|---|---|---|---|
| Linear Regression | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] |
| Gradient Boosting | [ ] | [ ] | [ ] |
| Gradient Boosting (tuned) | [ ] | [ ] | [ ] |

Best model: [model ka naam], R2 = [ ]

## Key Insights
- Price par sabse zyada asar: [top feature 1], [top feature 2], [top feature 3]
- Sabse mehnge regions (price per sqft): [region 1], [region 2], [region 3]
- [ek aur insight jo chart se mila]

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter/VS Code

## Files
- house_price.ipynb : poora code
- house_price_model.joblib : saved model
- model_columns.joblib : model ke columns

## Screenshots
(yahan 2 ya 3 charts ki images lagao, jaise feature importance aur actual vs predicted)
