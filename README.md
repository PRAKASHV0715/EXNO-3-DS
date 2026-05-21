## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
~~~
Name:PRAKASH V
REG:2122252030211
~~~
~~~
import numpy as np
from scipy import stats

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer

from category_encoders import BinaryEncoder
from category_encoders import TargetEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('data.csv')

print(df)

climate = ['Cold', 'Warm', 'Hot', 'Very Hot']

oe = OrdinalEncoder(categories=[climate])

df['Ord_1_Encoded'] = oe.fit_transform(df[['Ord_1']])

print(df)

le = LabelEncoder()

df['Ord_2_Label'] = le.fit_transform(df['Ord_2'])

print(df)

ohe = OneHotEncoder(sparse_output=False)

city_encoded = ohe.fit_transform(df[['City']])

city_columns = ohe.get_feature_names_out(['City'])

city_df = pd.DataFrame(city_encoded, columns=city_columns)

df = pd.concat([df, city_df], axis=1)

print(df)

be = BinaryEncoder(cols=['Ord_2'])

binary_df = be.fit_transform(df[['Ord_2']])

df = pd.concat([df, binary_df], axis=1)

print(df)

te = TargetEncoder(cols=['City'])

df['City_Target_Encoded'] = te.fit_transform(
    df['City'],
    df['Target']
)

print(df)

df.to_csv("Encoded_Data.csv", index=False)

df2 = pd.read_csv('Data_to_Transform.csv')

print(df2)

print(df2.skew())

df2["Highly Positive Skew_Log"] = np.log(
    df2["Highly Positive Skew"]
)

df2["Moderate Positive Skew_Reciprocal"] = np.reciprocal(
    df2["Moderate Positive Skew"]
)

df2["Highly Positive Skew_Sqrt"] = np.sqrt(
    df2["Highly Positive Skew"]
)

df2["Highly Positive Skew_Square"] = np.square(
    df2["Highly Positive Skew"]
)

df2["Highly Positive Skew_Boxcox"], parameters = stats.boxcox(
    df2["Highly Positive Skew"]
)

df2["Moderate Negative Skew_YeoJohnson"], parameters = stats.yeojohnson(
    df2["Moderate Negative Skew"]
)

qt = QuantileTransformer(output_distribution='normal')

df2["Moderate Negative Skew_QT"] = qt.fit_transform(
    df2[["Moderate Negative Skew"]]
)

df2["Highly Negative Skew_QT"] = qt.fit_transform(
    df2[["Highly Negative Skew"]]
)

sm.qqplot(df2["Moderate Negative Skew"], line='45')
plt.show()

sm.qqplot(df2["Moderate Negative Skew_QT"], line='45')
plt.show()

sm.qqplot(df2["Highly Negative Skew"], line='45')
plt.show()

sm.qqplot(np.reciprocal(df2["Moderate Negative Skew_QT"]), line='45')
plt.show()

sm.qqplot(df2["Highly Negative Skew_QT"], line='45')
plt.show()

sm.qqplot(np.abs(df2["Highly Negative Skew_QT"]), line='45')
plt.show()

sm.qqplot(np.log(df2["Highly Negative Skew_QT"]), line='45')
plt.show()

sm.qqplot(np.sqrt(df2["Moderate Negative Skew_QT"]), line='45')
plt.show()

df2.to_csv("Transformed_Data.csv", index=False)

~~~

<img width="936" height="603" alt="image" src="https://github.com/user-attachments/assets/1ae7a0cd-fc16-4545-8487-5e6cfe436c79" />

<img width="928" height="419" alt="image" src="https://github.com/user-attachments/assets/3d7ab1cc-48ed-4a3c-a68f-4383b6cf49a7" />

<img width="873" height="568" alt="image" src="https://github.com/user-attachments/assets/51da93e9-36c4-4f44-b22f-e2e9ef0cfe46" />

<img width="887" height="659" alt="image" src="https://github.com/user-attachments/assets/3aa91c5b-252c-48fd-8338-5a5713820576" />

<img width="798" height="579" alt="image" src="https://github.com/user-attachments/assets/ed8ab540-9915-4f09-9356-5ffb1bafca4e" />

<img width="898" height="655" alt="image" src="https://github.com/user-attachments/assets/8d5fb537-dec4-4021-8321-6556f7b0259b" />

<img width="923" height="451" alt="image" src="https://github.com/user-attachments/assets/622a888a-7dd0-4ca5-85e1-edb2fdfe54ab" />

<img width="673" height="599" alt="image" src="https://github.com/user-attachments/assets/bc46f933-981e-42aa-93d1-366efcd6ca55" />

<img width="803" height="604" alt="image" src="https://github.com/user-attachments/assets/96d90a01-b0a8-4b7b-935c-44bc37e1a410" />

<img width="911" height="582" alt="image" src="https://github.com/user-attachments/assets/06919630-cd0a-4b9f-9507-6ef153642769" />

<img width="926" height="573" alt="image" src="https://github.com/user-attachments/assets/6876108a-bff9-48b1-bfa5-8c7e8656b3ab" />

<img width="930" height="533" alt="image" src="https://github.com/user-attachments/assets/e1713795-3f3c-43b7-91f1-5b628217a980" />

<img width="861" height="254" alt="image" src="https://github.com/user-attachments/assets/d1b312d1-62c3-4b89-b6f2-a33e7ef19180" />

<img width="789" height="375" alt="image" src="https://github.com/user-attachments/assets/1846eba8-2cd5-4c22-a10b-e6a7f1b00ad0" />

<img width="832" height="421" alt="image" src="https://github.com/user-attachments/assets/a2501c18-d0d9-407a-b91e-2c6f0cce5e3e" />

<img width="806" height="401" alt="image" src="https://github.com/user-attachments/assets/42bf42d0-b9e3-48e7-8e8b-766531606292" />

<img width="933" height="425" alt="image" src="https://github.com/user-attachments/assets/25ebf7dd-0859-4d20-8a5c-98d71d42e169" />

<img width="936" height="376" alt="image" src="https://github.com/user-attachments/assets/c6a19fa8-7a4d-4087-9c44-035daa116ca6" />

<img width="931" height="428" alt="image" src="https://github.com/user-attachments/assets/6a3a96af-4296-4634-be9b-6c29c07a604f" />

<img width="937" height="657" alt="image" src="https://github.com/user-attachments/assets/eff36627-fcb0-4a29-8b96-669394bf9b82" />

<img width="932" height="631" alt="image" src="https://github.com/user-attachments/assets/dbb83191-4e1d-44b9-8893-f21acdbd9872" />

<img width="936" height="590" alt="image" src="https://github.com/user-attachments/assets/656230ce-2d12-45d3-b019-e54cf26e008f" />

<img width="926" height="568" alt="image" src="https://github.com/user-attachments/assets/345cf3e9-c736-4f69-b8e2-2fb78751bb7f" />

<img width="932" height="603" alt="image" src="https://github.com/user-attachments/assets/67fee838-b6dd-4283-a08b-5618bc6c5953" />

<img width="933" height="638" alt="image" src="https://github.com/user-attachments/assets/67a6b265-390b-4e02-a6d6-3f780828f4bb" />

<img width="932" height="516" alt="image" src="https://github.com/user-attachments/assets/39e05269-9faf-40aa-84da-604f18565e45" />

<img width="936" height="503" alt="image" src="https://github.com/user-attachments/assets/a83cfc9c-6447-4163-934e-0511537f63eb" />

<img width="805" height="531" alt="image" src="https://github.com/user-attachments/assets/be1c8648-edb0-47f4-9bfa-473f5dcb2176" />

# RESULT:
Thus, we have successfully performed Feature Encoding and Transformation process and saved the data to a file.
       
