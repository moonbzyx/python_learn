import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("My First app")

st.write("hello, world")

st.write("Here is our first attempt at using data to create a table")

st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]}))

df = pd.DataFrame({
    'first column': [5, 6, 7, 8],
    'second column': [50, 60, 70, 80]})

df
st.write(df)


chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

chart_data

st._legacy_line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x)

fig = plt.figure()
plt.title('Good funciton of sin')
plt.plot(x, y)

st.pyplot(fig)


# x = st.slider("x")
# y = x + 3
# y


# @st.cache
# def load_data():
#     df = pd.read_csv("data.csv")
#     df = df[['EVENT_TYPE','CREATE_TIME', 'COUNTY', 'LAT', 'LON']]
#     df.columns = ['event_type', 'time', 'county', 'lat', 'lon']
#     return df
