import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title('MRF Stock Price Visulization and Simple Linear Regression Model')

df = pd.read_csv('MRF.NS.csv')

#Describing Data
st.title('Data')
st.write(df.describe())

#Visualizations
st.title('Visualization')

# Graph 1: Line plot for 'Close'
st.write("Graph 1: Closing Prices Over Time")
st.line_chart(df.set_index('Date')['Close'])

# Graph 2: Bar plot for 'Open'
st.write("Graph 2: Open Price Over Time")
st.bar_chart(df.set_index('Date')['Open'])

# Graph 3: Scatter plot for 'Open' and 'Close'
st.write("Graph 3: Scatter Plot between Open and Close Prices")
st.scatter_chart(df, x='Open', y='Close')

# Graph 4: Histogram for 'Close Price'
st.write("Graph 4: Histogram for Close Price")
fig, ax = plt.subplots()
ax.hist(df['Close'], bins=100, edgecolor='black')
st.pyplot(fig)

# # Graph 5: Box plot for 'High'
# st.write("Graph 5: Box Plot for High Prices")
# fig=sns.boxplot(df['High'])
# st.pyplot(fig)

# # Graph 6: Pie Chart for 'Close Price' and 'Volume'
# st.write("Graph 6: Distribution of Volume by Close Price Range")
# volume_by_price_range = df.groupby('Price Range')['Volume'].sum()
# st.pie_chart(volume_by_price_range)

# # Graph 7: Violin plot for 'Adj Close'
# st.write("Graph 7: Violin Plot for Adj Close")
# st.violin_plot(df['Adj Close'])

"""#Filtering Data by applying GroupBy

Group by 'Series' and Calculate Maximum Volume:

This groups the data by the 'Series' column and calculates the maximum volume for each series.
"""

max_volume_by_high = df.groupby('High')['Volume'].max()
max_volume_by_high

"""Group by 'Year' and 'Month' and Calculate Total Volume:

This creates 'Year' and 'Month' columns from the 'Date' column, and then groups the data by year and month to calculate the total volume.
"""

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
total_turnover_by_year_month = df.groupby(['Year', 'Month'])['Volume'].sum()
total_turnover_by_year_month

"""Group by 'Year' and 'Month' and Calculate Average Close Price:

This creates 'Year' and 'Month' columns from the 'Date' column and groups the data by year and month, calculating the average close price for each period.
"""

avg_close_by_year_month = df.groupby(['Year', 'Month'])['Close'].mean()
avg_close_by_year_month

"""Group by 'Year' and Calculate Total Volume:

This creates a 'Year' column from the 'Date' column and groups the data by year, calculating the total volume for each year.
"""

total_volume_by_year = df.groupby('Year')['Volume'].sum()
total_volume_by_year

"""Group by 'Month' and Calculate Median Close Price:

This creates a 'Month' column from the 'Date' column and groups the data by month, calculating the median close price for each month.
"""

df['Month'] = df['Date'].dt.month
median_close_by_month = df.groupby('Month')['Close'].median()
median_close_by_month

"""#Filtering Data without using GroupBy

Filter for Close Prices above 1000 and below 10000:
"""

filter_condition = (df['Close'] > 1000) & (df['Close'] < 10000)
filtered_data = df[filter_condition]
filtered_data

"""Filter for Close Prices above 10000 and below 25000:"""

filter_condition = (df['Close'] >= 10000) & (df['Close'] < 25000)
filtered_data = df[filter_condition]
filtered_data

"""Filter to Calculate Highest Open Price:"""

highest_open_price = df[df['Open'] == df['Open'].max()]
highest_open_price

"""Filter to Calculate Highest Close Price:"""

highest_close_price = df[df['Close'] == df['Close'].max()]
highest_close_price

"""Filter to Calculate Highest Volume Traded:"""

highest_volume_traded = df[df['Volume'] == df['Volume'].max()]
highest_volume_traded

"""#Data PreProcessing

Finding Missing Values
"""

# df.isna

df.isna()

df.isnull().sum()

null_values = df[df['Close'].isnull()]

# Display the rows where 'Close' is null
print(null_values)

df.fillna(df.mean())

df.interpolate()

df=df.dropna()

"""#Applying Linear Regression"""
pip install scikit-learn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Data Preprocessing
# For simplicity, let's consider 'Close' as the target variable and 'Open' as the feature
X = df['Open'].values.reshape(-1, 1)  # Feature
y = df['Close'].values  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply Simple Linear Regression
# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Step 4: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 5: Visualize the results
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Linear Regression')
plt.title('Simple Linear Regression')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.legend()
plt.show()
