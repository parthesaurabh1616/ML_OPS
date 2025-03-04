#!/usr/bin/env python
# coding: utf-8

# # STEP 1: IMPORT ALL LAB'S

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# # STEP 2: LOAD DATASET

# In[2]:


sales_data = pd.read_csv("C://Users/saura/Downloads/Sales_Data.csv")
iod_data = pd.read_csv("C://Users/saura/Downloads/IOD.csv")
obd_data = pd.read_csv("C://Users/saura/Downloads/OBD.csv")


# # STEP 3: INSPECT AND CLEAN THE DATA

# In[5]:


print(sales_data.info())
print(iod_data.info())
print(obd_data.info())

# Check for missing values
print(sales_data.isnull().sum())
print(iod_data.isnull().sum())
print(obd_data.isnull().sum())


# # FIXES:

# In[4]:


date_columns = ["Invoice_Date", "Billing_Date", "IOD_Date", "OBD_Date", "Pick_Date"]
for col in date_columns:
    for df in [sales_data, iod_data, obd_data]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')


# In[6]:


iod_data["Delivery_Days"].fillna(iod_data["Delivery_Days"].median(), inplace=True)


# # STEP 4: HANDLE DUPLICATES IN (OBD_NO)

# In[7]:


sales_grouped = sales_data.groupby("OBD_No", as_index=False).agg({
    "Product": "first",
    "Month": "first",
    "Supplying_Plant_Code": "first",
    "Channel": "first",
    "Sales_Office_ID": "first",
    "Invoice_No": "first",
    "Invoice_Date": "first",
    "Ship_to_Code_City": "first",
    "QTY": "sum",
    "Sales_Value": "sum",
    "Supplying_Plant_City": "first",
    "Sales_Office_City": "first",
})


# # STEP 5: MERGE THE DATASETS

# In[8]:


# Merge sales_data with iod_data on Invoice_No:
merged_sales_iod = sales_grouped.merge(iod_data, on="Invoice_No", how="left")


# In[9]:


# Merge with obd_data on OBD_No:
merged_sales_iod.rename(columns={"OBD_No_x": "OBD_No"}, inplace=True)  # Resolve naming conflicts if needed
merged_sales_iod_obd = merged_sales_iod.merge(obd_data, on="OBD_No", how="left")


# In[10]:


# Fill missing values in OBD_Date using Invoice_Date:
merged_sales_iod_obd["OBD_Date"].fillna(merged_sales_iod_obd["Invoice_Date"], inplace=True)


# # STEP 6: FEATURE ENGINEERING

# In[11]:


merged_sales_iod_obd["Delivery_Delayed"] = merged_sales_iod_obd["Delivery_Days"].apply(lambda x: 1 if x > 5 else 0)

merged_sales_iod_obd["Time_To_Invoice"] = (merged_sales_iod_obd["Billing_Date"] - merged_sales_iod_obd["Invoice_Date"]).dt.days
merged_sales_iod_obd["Time_To_IOD"] = (merged_sales_iod_obd["IOD_Date"] - merged_sales_iod_obd["Billing_Date"]).dt.days
merged_sales_iod_obd["Total_Processing_Time"] = (merged_sales_iod_obd["IOD_Date"] - merged_sales_iod_obd["Invoice_Date"]).dt.days


# # STEP 7: TRAIN MACHINE LEARNING MODEL

# # Prepare Data for Training

# In[12]:


features = ["Supplying_Plant_Code", "Sales_Office_ID", "QTY", "Sales_Value", "Time_To_Invoice", "Time_To_IOD", "Total_Processing_Time"]
target = "Delivery_Delayed"

# Drop rows with NaN values in features
cleaned_data = merged_sales_iod_obd.dropna(subset=features + [target])

X = cleaned_data[features]
y = cleaned_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# # Train a Random Forest Classifier

# In[13]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# # Evaluate Model Performance

# In[14]:


y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# # STEP 8: DATA VISUALIZATION

# # Feature Importance

# In[15]:


importances = rf_model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_indices], align="center")
plt.xticks(range(len(importances)), [features[i] for i in sorted_indices], rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Predicting Delivery Delay")
plt.show()


# # Delivery Delay Trends Over Time

# In[16]:


delay_trend = merged_sales_iod_obd.groupby(merged_sales_iod_obd["Invoice_Date"].dt.to_period("M"))["Delivery_Delayed"].mean()

plt.figure(figsize=(12, 6))
sns.lineplot(x=delay_trend.index.astype(str), y=delay_trend.values, marker="o")
plt.xticks(rotation=45)
plt.xlabel("Month-Year")
plt.ylabel("Percentage of Delayed Deliveries")
plt.title("Delivery Delay Trends Over Time")
plt.show()


# # Top 15 Products with the Highest Delays

# In[17]:


plt.figure(figsize=(12, 6))
product_delay = merged_sales_iod_obd.groupby("Product")["Delivery_Delayed"].mean().sort_values(ascending=False).head(15)

sns.barplot(x=product_delay.values, y=product_delay.index, palette="viridis")
plt.xlabel("Percentage of Delayed Deliveries")
plt.ylabel("Product")
plt.title("Top 15 Products with Highest Delivery Delays")
plt.show()


# # Delays by City

# In[18]:


plt.figure(figsize=(12, 6))
city_delay = merged_sales_iod_obd.groupby("Sales_Office_City")["Delivery_Delayed"].mean().sort_values(ascending=False).head(15)

sns.barplot(x=city_delay.values, y=city_delay.index, palette="coolwarm")
plt.xlabel("Percentage of Delayed Deliveries")
plt.ylabel("Sales Office City")
plt.title("Top 15 Cities with Highest Delivery Delays")
plt.show()


# # STEP 9: SAVE DATA

# In[19]:


merged_sales_iod_obd.to_csv("Processed_Sales_Data.csv", index=False)


# In[21]:


from IPython.display import FileLink

FileLink("Processed_Sales_Data.csv")

# %%
