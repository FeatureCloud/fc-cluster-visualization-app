import json

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    html.H1('Visualization App'),
    dcc.Tabs(id="tabs-ct", value='tab-distances', children=[
        dcc.Tab(label='Confounders', value='tab-confounders'),
        dcc.Tab(label='Distances', value='tab-distances'),
        dcc.Tab(label='Clustering Quality', value='tab-clustering-quality'),
        dcc.Tab(label='Scree plot', value='tab-scree-plot'),
        dcc.Tab(label='Scatter plot', value='tab-scatter-plot'),
    ]),
    html.Div(id='tabs-content-ct')
])

distance_df = pd.read_csv("data/distanceMatrix.csv", delimiter=" ", skiprows=0, index_col=0) #Code A

@app.callback(Output('tabs-content-ct', 'children'),
              Input('tabs-ct', 'value'))

def render_content(tab):
    if tab == 'tab-confounders':
        return renderConfounders()
    elif tab == 'tab-distances':
        return renderDistances()
    elif tab == 'tab-clustering-quality':
        return renderClusteringQuality()
    elif tab == 'tab-scree-plot':
        return renderScreePlot()
    elif tab == 'tab-scatter-plot':
        return renderScatterPlot()


def renderConfounders():
    return html.H1("Confounders")

def renderDistances():
    return html.Div([
        html.P("Labels included:"),
        dcc.Dropdown(
            id='labels',
            options=[{'label': x, 'value': x}
                     for x in distance_df.columns],
            value=distance_df.columns.tolist(),
            multi=True,
        ),
        dcc.Graph(id="graph"),
    ])

@app.callback(
    Output("graph", "figure"),
    [Input("labels", "value")])
def filter_heatmap(cols):
    data = {
        'z': distance_df[cols].values.tolist(),
        'x': distance_df[cols].columns.tolist(),
        'y': distance_df[cols].index.tolist()
    }
    layout = go.Layout(
        title='Distance matrix',
    )
    fig = go.Figure(data=go.Heatmap(data), layout=layout)
    return fig

def renderClusteringQuality():
    return html.H1("Clustering Quality")

def renderScreePlot():
    return html.H1("Scree plot")

def renderScatterPlot():
    df = pd.DataFrame({
        "x": [1, 2, 1, 2],
        "y": [1, 2, 3, 4],
        "customdata": [1, 2, 3, 4],
        "fruit": ["apple", "apple", "orange", "orange"]
    })

    fig = px.scatter(df, x="x", y="y", color="fruit", custom_data=["customdata"])

    fig.update_layout(clickmode='event+select')

    fig.update_traces(marker_size=20)

    return html.Div([
        dcc.Graph(
            id='basic-interactions',
            figure=fig
        ),

        html.Div(className='row', children=[
            html.Div([
                dcc.Markdown("""
                    **Hover Data**

                    Mouse over values in the graph.
                """),
                html.Pre(id='hover-data', style=styles['pre'])
            ], className='three columns'),

            html.Div([
                dcc.Markdown("""
                    **Click Data**

                    Click on points in the graph.
                """),
                html.Pre(id='click-data', style=styles['pre']),
            ], className='three columns'),

            html.Div([
                dcc.Markdown("""
                    **Selection Data**

                    Choose the lasso or rectangle tool in the graph's menu
                    bar and then select points in the graph.

                    Note that if `layout.clickmode = 'event+select'`, selection data also
                    accumulates (or un-accumulates) selected data if you hold down the shift
                    button while clicking.
                """),
                html.Pre(id='selected-data', style=styles['pre']),
            ], className='three columns'),

            html.Div([
                dcc.Markdown("""
                    **Zoom and Relayout Data**

                    Click and drag on the graph to zoom or click on the zoom
                    buttons in the graph's menu bar.
                    Clicking on legend items will also fire
                    this event.
                """),
                html.Pre(id='relayout-data', style=styles['pre']),
            ], className='three columns')
        ])
    ])


@app.callback(
    Output('hover-data', 'children'),
    Input('basic-interactions', 'hoverData'))
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    Output('click-data', 'children'),
    Input('basic-interactions', 'clickData'))
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


@app.callback(
    Output('selected-data', 'children'),
    Input('basic-interactions', 'selectedData'))
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)


@app.callback(
    Output('relayout-data', 'children'),
    Input('basic-interactions', 'relayoutData'))
def display_relayout_data(relayoutData):
    return json.dumps(relayoutData, indent=2)


if __name__ == '__main__':
    app.run_server(debug=True)