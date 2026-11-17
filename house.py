import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv("house.csv")
df.head()
print("done head")

df.info()
print("info done ")

print(df.shape)
df["price_lakh"] = np.where(df["price_unit"] == "Cr", df["price"] * 100, df["price"])
df[["price", "price_unit", "price_lakh"]].head(10)

df=df.drop_duplicates()

df["price_per_sqft"] = (df["price_lakh"] * 100000) / df["area"]
df["price_per_sqft"].describe()

df = df.drop(columns=["price", "price_unit"])
df.head()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(y=df["area"], ax=axes[0])
sns.boxplot(y=df["price_lakh"], ax=axes[1])


sns.boxplot(y=df["price_per_sqft"], ax=axes[2])
plt.tight_layout()
plt.show()

print(df.sort_values("price_per_sqft").head(5))
print(df.sort_values("price_per_sqft").tail(5))

low = df["price_per_sqft"].quantile(0.01)
high = df["price_per_sqft"].quantile(0.99)
print("Low cutoff:", low, "High cutoff:", high)

print("Before:", df.shape)
df = df[(df["price_per_sqft"] >= low) & (df["price_per_sqft"] <= high)]
print("After:", df.shape)

df["area_per_bhk"] = df["area"] / df["bhk"]
print(df["area_per_bhk"].describe())

df = df[(df["area_per_bhk"] >= 150) & (df["area_per_bhk"] <= 1500)]
print("After:", df.shape)




fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(y=df["area"], ax=axes[0])
sns.boxplot(y=df["price_lakh"], ax=axes[1])
sns.boxplot(y=df["price_per_sqft"], ax=axes[2])
plt.tight_layout()
plt.show()

df=df.drop(columns=["locality"])
df.head()

region_count=df['region'].value_counts()
print(region_count.head(15))
print("region with less than 100 rows ",(region_count < 100).sum())

rare_region=region_count[region_count < 100].index
df["region"]=df["region"].replace(rare_region,"other")
print(df["region"].unique())


df["area_per_bhk"] = df["area"] / df["bhk"]
df.head()

plt.figure(figsize=(8, 4))
sns.histplot(df["price_lakh"], bins=50)
plt.title("Price Distribution (in lakh)")
plt.xlabel("Price (lakh)")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="area", y="price_lakh", data=df.sample(3000, random_state=1), alpha=0.5)
plt.title("Area vs Price")
plt.show()

top_regions = df.groupby("region")["price_per_sqft"].mean().sort_values(ascending=False).head(10)
print(top_regions)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_regions.values, y=top_regions.index)
plt.title("Top 10 Regions by Avg Price per Sqft")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="bhk", y="price_lakh", data=df[df["bhk"] <= 5])
plt.title("BHK vs Price")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
df.groupby("type")["price_lakh"].mean().plot(kind="bar", ax=axes[0], title="Avg Price by Type")
df.groupby("status")["price_lakh"].mean().plot(kind="bar", ax=axes[1], title="Avg Price by Status")
df.groupby("age")["price_lakh"].mean().plot(kind="bar", ax=axes[2], title="Avg Price by Age")
plt.tight_layout()
plt.show()

num_cols = ["bhk", "area", "price_lakh", "price_per_sqft", "area_per_bhk"]
plt.figure(figsize=(7, 5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

X = df.drop(columns=["price_lakh", "price_per_sqft"])
y = df["price_lakh"]

print(X.columns.tolist())
print(X.shape, y.shape)

X = pd.get_dummies(X, columns=["type", "region", "status", "age"], drop_first=True)
print(X.shape)
X.head()

X = X.astype(int)
X.dtypes.value_counts()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

result = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
result.head(10)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE (lakh):", round(mae, 2))
print("RMSE (lakh):", round(rmse, 2))
print("R2 score:", round(r2, 3))
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
plt.xlabel("Actual Price (lakh)")
plt.ylabel("Predicted Price (lakh)")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
from sklearn.ensemble import HistGradientBoostingRegressor

gb = HistGradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

models = {
    "Linear Regression": y_pred,
    "Random Forest": rf_pred,
    "Gradient Boosting": gb_pred,
}

rows = []
for name, pred in models.items():
    rows.append({
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, pred)), 2),
        "R2": round(r2_score(y_test, pred), 3),
    })

comparison = pd.DataFrame(rows)
comparison

plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
plt.xlabel("Actual Price (lakh)")
plt.ylabel("Predicted Price (lakh)")
plt.title("Random Forest: Actual vs Predicted")
plt.show()

from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_iter": [100, 200, 300],
    "max_depth": [None, 6, 10, 15],
    "min_samples_leaf": [10, 20, 40],
}

search = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=6,
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=1,
)
search.fit(X_train, y_train)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

print("Best settings:", search.best_params_)
print("Best CV R2:", round(search.best_score_, 3))

best_gb = search.best_estimator_
best_pred = best_gb.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, best_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, best_pred)), 2))
print("R2:", round(r2_score(y_test, best_pred), 3))

importance = pd.Series(rf.feature_importances_, index=X_train.columns)
top = importance.sort_values(ascending=False).head(15)
print(top)

plt.figure(figsize=(9, 6))
sns.barplot(x=top.values, y=top.index)
plt.title("Top 15 Features Affecting Price")
plt.show()
print(df["type"].unique())
print(df["status"].unique())
print(df["age"].unique())
print(df["region"].unique()[:20])
def predict_price(bhk, area, type_, region, status, age):
    row = pd.DataFrame(0.0, index=[0], columns=X_train.columns)

    row["bhk"] = bhk
    row["area"] = area
    row["area_per_bhk"] = area / bhk

    for col, value in [("type", type_), ("region", region), ("status", status), ("age", age)]:
        col_name = f"{col}_{value}"
        if col_name in row.columns:
            row[col_name] = 1
        elif col == "region" and "region_Other" in row.columns:
            row["region_Other"] = 1

    price = best_gb.predict(row)[0]
    return round(price, 2)
price = predict_price(bhk=2, area=900, type_="Apartment", region="Andheri East", status="Ready to move", age="New")
print("Predicted price (lakh):", price)
print("Predicted price (crore):", round(price / 100, 2))

import joblib

joblib.dump(best_gb, "house_price_model.joblib")
joblib.dump(list(X_train.columns), "model_columns.joblib")







