import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np

import dash_table

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_february_us_airport_traffic.csv')

# Subset dataframe to show some specific columns in dash web app
df1 = df[['city','state','country','lat','long','cnt']]

app = dash.Dash(external_stylesheets=['https://codepen.io/amyoshino/pen/jzXypZ.css'])

app.title = 'MapBox Map'

# API keys and datasets
mapbox_access_token = 'Paste your Mapbox Access token here inside qutation'

# Find Lat Long center
lat_center = sum(df['lat'])/len(df['lat'])
long_center = sum(df['long'])/len(df['long'])

layout_map = dict(
    autosize=True,
    height=500,
    weidth=100,
    font=dict(color="#191A1A"),
    titlefont=dict(color="#191A1A", size='14'),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    hovermode="closest",
    plot_bgcolor='#fffcfc',
    paper_bgcolor='#fffcfc',
    legend=dict(font=dict(size=10), orientation='h'),
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="outdoors",
        center=dict(
            lon = long_center,
            lat = lat_center
        ),
        zoom=2,
    )
)

app.layout = html.Div(
    html.Div([
        html.Div([
            html.H1(children='Plot Lat Long using Mapbox in Dash'),

            html.H6(children='''
                Plotting all airports in map using respective lat long value
            '''),
   html.Div(id='my-div'),
        ], className = 'row'),

        html.Br(),

        html.Div([
            html.Div([
        dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in df1.columns],
                        data=df1.loc[:14,].to_dict('records'),
                    ),
            ], className = 'six columns'),
    


            html.Div([
    dcc.Graph(
     id='MapPlot',
      figure={
       "data": [{
        "type": "scattermapbox",
        "lat": list(df.lat),
        "lon": list(df.long),
        "hoverinfo": "text",
        "hovertext": [["Lat: {} 
Long: {} 
Count: {}".format(i,j,k)]
            for i,j,k in zip(df['lat'], df['long'],df['cnt'])],
        "mode": "markers",
        "name": list(df['state']),
        "marker": {
         "size": 15,
         "opacity": 0.7,
         "color": '#F70F0F'
        }
       }],
       "layout": layout_map
      }
    ),
            ], className = 'six columns')
        ], className = 'row')

    ])
)

if __name__ == '__main__':
    app.run_server(debug=False)