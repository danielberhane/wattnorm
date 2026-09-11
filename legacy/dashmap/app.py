# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd

app = dash.Dash(__name__)

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

df = pd.read_csv('Geothermals.csv')
df['text'] = df['Name'] + ', ' + df['State']

fig = go.Figure(data=go.Scattergeo(
    lon=df['Lon_84'],
    lat=df['Lat_84'],
    text=df['text'],
    mode='markers',
    marker_color=df['Temp_C_ML']
))

fig.update_layout(
    geo_scope='usa'
)

app.layout = html.Div(children=[
    html.H1(children='Identified Geothermal Systems of the Western USA'),
    html.Div(children='''
        This data was provided by the USGS.
    '''),

    dcc.Graph(
        id='example-map',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True, port=1234)
