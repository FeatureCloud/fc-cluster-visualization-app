import json

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, ALL
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import numpy as np
from plotly.subplots import make_subplots
from plotly.tools import DEFAULT_PLOTLY_COLORS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__)#, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    html.H2('Visualization App'),
    dcc.Tabs(id="tabs-ct", value='tab-confounders', children=[
        dcc.Tab(label='Confounders', value='tab-confounders'),
        dcc.Tab(label='Distances', value='tab-distances'),
        dcc.Tab(label='Clustering Quality', value='tab-clustering-quality'),
        dcc.Tab(label='Scree plot', value='tab-scree-plot'),
        # dcc.Tab(label='Scatter plot', value='tab-scatter-plot'),
    ]),
    html.Div(id='tabs-content-ct', style={'width': '75%', 'margin': '0 auto'})
])

distance_df = pd.read_csv("data/distanceMatrix.csv", delimiter=" ", skiprows=0, index_col=0)
confounding_meta = pd.read_csv(f'data/confoundingData.meta', delimiter=";", skiprows=0)


@app.callback(
    Output('tabs-content-ct', 'children'),
    Input('tabs-ct', 'value')
)
def render_content(tab):
    if tab == 'tab-confounders':
        return renderConfounders()
    elif tab == 'tab-distances':
        return renderDistances()
    elif tab == 'tab-clustering-quality':
        return renderClusteringQuality()
    elif tab == 'tab-scree-plot':
        return renderScreePlot()
    # elif tab == 'tab-scatter-plot':
    #     return renderScatterPlot()


def renderConfounders():
    return html.Div(
        [
            html.Div(
                [
                    html.Span('K', style={'float': 'left', 'width': '15%'}),
                    html.Span(dcc.Dropdown([2, 3], 2, id='k-confounders', style={'width': '75%', 'float': 'left'}))
                ],
                style={'width': '100%', 'display': 'block', 'margin-top': '20px'}
            ),
            html.Div(getConfoundingFactorsFilter('confounders'), style={'height': '16vh'}),
            dcc.Graph(id='confounders-scatter', style={'width': '150vh', 'height': '80vh'})
        ]
    )


@app.callback(
    Output('confounders-scatter', 'figure'),
    Input('k-confounders', 'value'),
    Input({'type': 'filter-checklist-confounders', 'index': ALL}, 'value'),
    Input({'type': 'filter-range-slider-confounders', 'index': ALL}, 'value'),
)
def filter_k_confounders(value, checklist_values, range_values):
    # to be changed later for an automatic counting
    confounding_df = pd.read_csv(f'data/all_confounders_{value}.csv', delimiter=";", skiprows=0)

    # filter base dataframe
    index_list = filterDataframeOnCounfoundingFactors(confounding_df, checklist_values, range_values)
    confounding_df = confounding_df[confounding_df.index.isin(index_list)]

    cluster_values_list = confounding_df.cluster.unique()
    nr_of_confounding_factors = len(confounding_meta.index)
    nr_rows = nr_of_confounding_factors + value
    nr_cols = nr_of_confounding_factors

    specs, subplot_titles = getSpecsForMatrix(nr_rows, nr_cols)
    fig = make_subplots(
        rows=nr_rows,
        cols=nr_cols,
        specs=specs,
        subplot_titles=subplot_titles
    )
    for i in cluster_values_list:
        color = DEFAULT_PLOTLY_COLORS[i]
        df = confounding_df[confounding_df['cluster'] == i]
        scatter_plot = go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers',
            name=f'Cluster {i}',
            marker={
                "size": 10,
                "color": color,
            }
        )
        fig.append_trace(scatter_plot, row=1, col=1)
        path = confidence_ellipse(df['x'],
                                  df['y'])
        fig.add_shape(
            type='path',
            path=path,
            line={'dash': 'dot'},
            line_color=color,
            fillcolor=color,
            opacity=0.15,
            row=1,
            col=1
        )

    for i in cluster_values_list:
        color = DEFAULT_PLOTLY_COLORS[i]
        df = confounding_df[confounding_df['cluster'] == i]
        for j in range(0, len(confounding_meta.index)):
            col = confounding_meta.iloc[j]['name']
            data_type = confounding_meta.iloc[j]['data_type']
            if data_type == 'continuous':
                # add histogram
                bar_continuous = go.Histogram(
                    x=df[col],
                    marker={'color': color},
                    hovertemplate=col.capitalize() + ' group: %{x}<br>Count: %{y}',
                    showlegend=False,
                )
                fig.add_trace(bar_continuous, row=i + nr_cols, col=j + 1)
            elif data_type == 'discrete':
                # add pie chart
                pie_values_list = []
                discrete_val_list = confounding_df[col].unique()

                for discrete_val in discrete_val_list:
                    pie_values_list.append(df[df[col] == discrete_val].count()[col])
                fig.add_trace(go.Pie(labels=discrete_val_list, values=pie_values_list, showlegend=False),
                              row=i + nr_cols, col=j + 1)
    return fig


def renderDistances():
    return html.Div([
        dcc.Graph(id="distance_graph"),
        html.Div(
            children=getConfoundingFactorsFilter('distance'),
            style={'margin': '0 auto', 'width': '70%'}
        )
    ])


@app.callback(
    Output("distance_graph", "figure"),
    Input({'type': 'filter-checklist-distance', 'index': ALL}, 'value'),
    Input({'type': 'filter-range-slider-distance', 'index': ALL}, 'value'),
)
def filter_heatmap(checklist_values, range_values):
    confounding_df = pd.read_csv(f'data/all_confounders_3.csv', delimiter=";", skiprows=0)
    index_list = filterDataframeOnCounfoundingFactors(confounding_df, checklist_values, range_values)
    df = distance_df[distance_df.index.isin(index_list)]
    data = {
        'z': df.values.tolist(),
        'x': df.columns.tolist(),
        'y': df.index.tolist()
    }
    layout = go.Layout(
        title='Distance matrix',
        xaxis={
            "title": "",
            "showticklabels": False,
        },
        yaxis={
            "title": "",
            "showticklabels": False,
        },
    )
    fig = go.Figure(data=go.Heatmap(data), layout=layout)
    return fig


def filterDataframeOnCounfoundingFactors(confounding_df, checklist_values, range_values):
    confounding_length = len(confounding_meta.index)
    # Filter data based on active filters
    checklist_index = range_index = 0
    for j in range(0, confounding_length):
        col = confounding_meta.iloc[j]["name"]
        data_type = confounding_meta.iloc[j]['data_type']
        if data_type == 'continuous':
            range_list = range_values[range_index]
            confounding_df = confounding_df.loc[confounding_df[col].between(range_list[0], range_list[1])]
            range_index += 1
        elif data_type == 'discrete':
            checklist = checklist_values[checklist_index]
            confounding_df = confounding_df.loc[confounding_df[col].isin(checklist)]
            checklist_index += 1
    return confounding_df.index.tolist()


def renderClusteringQuality():
    return html.Div([
        html.P("K:"),
        dcc.Dropdown([2, 3], 2, id='k-labels'),
        dcc.Graph(id="cluster_quality_graph"),
    ])


@app.callback(
    Output('cluster_quality_graph', 'figure'),
    Input('k-labels', 'value'))
def filter_k_label(value):
    df_silhouette = pd.read_csv(f'data/results/K_{str(value)}/silhouette.csv', delimiter=';')
    df_silhouette = df_silhouette.sort_values(["cluster", "y"], ascending=(True, False)).reset_index()
    avg_value = "{:.2f}".format(df_silhouette['y'].mean())
    cluster_values_list = df_silhouette.cluster.unique()

    fig = go.Figure()
    for i in cluster_values_list:
        fig.add_trace(
            go.Bar(
                y=df_silhouette[df_silhouette['cluster'] == i]['y'],
                x=df_silhouette[df_silhouette['cluster'] == i].index,
                name=f'Cluster {i}',
                marker={
                    "line": {
                        "width": 0,
                    },
                }
            )
        )
    # Add avg line on top
    fig.add_shape(
        type="line",
        x0=0, y0=avg_value, x1=df_silhouette['x'].max(), y1=avg_value,
        line=dict(
            color="Red",
            width=2,
            dash="dashdot",
        )
    )
    fig.update_layout(
        title=f'Clusters silhouette plot<br>Average silhouette width: {str(avg_value)}',
        xaxis={
            "title": "",
            "showticklabels": False,
        },
        yaxis={"title": "Silhouette width Si"},
        bargap=0.0,
        showlegend=True,
        legend={
            "title": "Clusters",
        },
    )
    return fig


def renderScreePlot():
    # define URL where dataset is located
    url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/USArrests.csv"

    # read in data
    data = pd.read_csv(url)

    # define columns to use for PCA
    df = data.iloc[:, 1:5]

    # define scaler
    scaler = StandardScaler()

    # create copy of DataFrame
    scaled_df = df.copy()

    # created scaled version of DataFrame
    scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)

    # define PCA model to use
    pca = PCA(n_components=4)

    # fit PCA model to data
    pca.fit(scaled_df)
    PC_values = np.arange(pca.n_components_) + 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        mode='lines+markers',
        x=PC_values,
        y=pca.explained_variance_ratio_,
        marker={
            "size": 10,
            "symbol": "circle-open",
        }
    ))
    fig.update_layout(
        title="Scree plot",
        xaxis_title="Component number",
        yaxis_title="Eigenvalue",

    )
    return html.Div([
        dcc.Graph(
            id='scree-plot',
            figure=fig
        )])


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


# @app.callback(
#     Output('hover-data', 'children'),
#     Input('basic-interactions', 'hoverData'))
# def display_hover_data(hoverData):
#     return json.dumps(hoverData, indent=2)
#
#
# @app.callback(
#     Output('click-data', 'children'),
#     Input('basic-interactions', 'clickData'))
# def display_click_data(clickData):
#     return json.dumps(clickData, indent=2)
#
#
# @app.callback(
#     Output('selected-data', 'children'),
#     Input('basic-interactions', 'selectedData'))
# def display_selected_data(selectedData):
#     return json.dumps(selectedData, indent=2)
#
#
# @app.callback(
#     Output('relayout-data', 'children'),
#     Input('basic-interactions', 'relayoutData'))
# def display_relayout_data(relayoutData):
#     return json.dumps(relayoutData, indent=2)


def confidence_ellipse(x, y, n_std=1.96, size=100):
    """
        Get the covariance confidence ellipse of *x* and *y*.
        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.
        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.
        size : int
            Number of points defining the ellipse
        Returns
        -------
        String containing an SVG path for the ellipse

        References (H/T)
        ----------------
        https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
        https://community.plotly.com/t/arc-shape-with-path/7205/5
        """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack([ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)])

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    # calculating the stdandard deviation of y ...
    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)

    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4)],
                                [-np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    scale_matrix = np.array([[x_scale, 0],
                             [0, y_scale]])
    ellipse_coords = ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix

    path = f'M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}'
    for k in range(1, len(ellipse_coords)):
        path += f'L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}'
    path += ' Z'
    return path

def getSpecsForMatrix(rows, cols):
    specs = []
    subplot_titles = []
    for i in range(1, rows + 1):
        current_specs_row = []
        if i == 1:
            current_specs_row.append({'rowspan': cols, 'colspan': cols})
            subplot_titles.append("Confidence ellipsis")
            for j in range(1, cols):
                current_specs_row.append(None)
        elif i <= cols:
            for j in range(1, cols + 1):
                current_specs_row.append(None)
        else:
            for j in range(0, len(confounding_meta.index)):
                current_specs_row.append(
                    {'type': 'xy' if confounding_meta.iloc[j]['data_type'] == 'continuous' else 'pie'})
                subplot_titles.append(confounding_meta.iloc[j]['name'].capitalize())
        specs.append(current_specs_row)
    return specs, subplot_titles


def getConfoundingFactorsFilter(id_pre_tag):
    html_elem_list = []
    confounding_df = pd.read_csv(f'data/all_confounders_3.csv', delimiter=";", skiprows=0)
    confounding_length = len(confounding_meta.index)
    for j in range(0, confounding_length):
        col = confounding_meta.iloc[j]["name"]
        data_type = confounding_meta.iloc[j]['data_type']
        if data_type == 'continuous':
            # add range slider
            col_min = confounding_df[col].min()
            col_max = confounding_df[col].max()
            html_elem_list.append(
                html.Div(
                    [
                        html.Span(col.capitalize(), style={'float': 'left', 'width': '15%'}),
                        html.Span(
                            dcc.RangeSlider(
                                col_min, col_max, value=[col_min, col_max],
                                id={
                                    'type': f'filter-range-slider-{id_pre_tag}',
                                    'index': j
                                }
                            ),
                            style={'width': '75%', 'float': 'left'}
                        ),
                    ],
                    style={'width': '100%', 'display': 'block'}
                )
            )
        elif data_type == 'discrete':
            # add checklist
            discrete_val_list = confounding_df[col].unique()
            html_elem_list.append(
                html.Div(
                    [
                        html.Span(col.capitalize(), style={'float': 'left', 'width': '15%'}),
                        html.Span(
                            dcc.Checklist(
                                discrete_val_list, discrete_val_list, inline=True,
                                id={
                                    'type': f'filter-checklist-{id_pre_tag}',
                                    'index': j
                                }
                            ),
                            style={'width': '75%', 'float': 'left'}
                        ),
                    ],
                    style={'width': '100%', 'display': 'block'}
                )
            )
    return html_elem_list


if __name__ == '__main__':
    app.run_server(debug=True)
