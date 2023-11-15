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
plt.hist(df['Close'], bins=100, edgecolor='black')
st.pyplot()

# Graph 5: Box plot for 'High'
st.write("Graph 5: Box Plot for High Prices")
st.box_plot(df['High'])

# Graph 6: Pie Chart for 'Close Price' and 'Volume'
st.write("Graph 6: Distribution of Volume by Close Price Range")
volume_by_price_range = df.groupby('Price Range')['Volume'].sum()
st.pie_chart(volume_by_price_range)

# Graph 7: Violin plot for 'Adj Close'
st.write("Graph 7: Violin Plot for Adj Close")
st.violin_plot(df['Adj Close'])
