import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Exited']
df_churn = pd.read_csv("Chum_Modeling.csv", usecols=cols)
# CreditScore в Германии и Франции на основе возраста
sample = df_churn.sample(n=200, random_state=42)
fig, ax = plt.subplots()
plt.title("France vs Germany", fontsize=14)
ax.scatter(x=sample[sample.Geography == 'France']['CreditScore'], y=sample[sample.Geography == 'France']['Age'])
ax.scatter(x=sample[sample.Geography == 'Germany']['CreditScore'], y=sample[sample.Geography == 'Germany']['Age'])
ax.legend(labels=['France', 'Germany'], loc='lower left', fontsize=12)
ax.set_xlabel("CreditScore $")  # Подпись оси x
ax.set_ylabel("Age")  # Подпись оси y
plt.show()

# Кол-во покупок в страннах и кол-во покупок конкретных товаров
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8,5))
countries = df_churn.Geography.value_counts()
products = df_churn.NumOfProducts.value_counts()
ax1.bar(x=countries.index, height=countries.values)
ax1.set_title("Countries", fontsize=12)
ax2.bar(x=products.index, height=products.values)
ax2.set_title("Number of Products", fontsize=12)
plt.show()

# Разделим данные на две группы: Франция и Германия
credit_france = df_churn[df_churn['Geography'] == 'France']['CreditScore']
credit_germany = df_churn[df_churn['Geography'] == 'Germany']['CreditScore']
str1 = df_churn["Age"]
str2 = df_churn["CreditScore"]
# Проведем t-тест
corr, _ = pearsonr(str1, str2)
print("Коэффициент корелляции: ", corr)
# Выведем результаты теста
print(df_churn.corr(method='pearson', min_periods=1, numeric_only=True))
