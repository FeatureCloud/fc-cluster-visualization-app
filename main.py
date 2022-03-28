import json
import os

from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, ALL
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import numpy as np
from plotly.subplots import make_subplots
from plotly.tools import DEFAULT_PLOTLY_COLORS

app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO], title='FeatureCloud Visualization App')

DISTANCE_DF = []
CONFOUNDING_META = []
DF_SCREE_PLOT = []
K_VALUES =[]
DATAFRAMES_BY_CLUSTER = []
DF_SILHOUETTE = []
DELIMITER = ';'
DATA_DIR = "./data"
RESULT_DIR = f'{DATA_DIR}/results'

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    html.H2('Cluster Visualization', className='fc-header'),
    dcc.Tabs(id="tabs-ct", value='tab-confounders', children=[
        dcc.Tab(label='Confounders', value='tab-confounders'),
        dcc.Tab(label='Distances', value='tab-distances'),
        dcc.Tab(label='Clustering Quality', value='tab-clustering-quality'),
        dcc.Tab(label='Scree plot', value='tab-scree-plot'),
        dcc.Tab(label='Scatter plot', value='tab-scatter-plot'),
    ]),
    html.Div(id='tabs-content-ct', style={'width': '75%', 'margin': '0 auto'})
])


def assembleDataframes():
    global DISTANCE_DF, CONFOUNDING_META, DATAFRAMES_BY_CLUSTER, DF_SILHOUETTE, DF_SCREE_PLOT, K_VALUES
    DISTANCE_DF = pd.read_csv(f'{DATA_DIR}/distanceMatrix.csv', delimiter=DELIMITER, skiprows=0, index_col=0)
    CONFOUNDING_META = pd.read_csv(f'{DATA_DIR}/confoundingData.meta', delimiter=DELIMITER, skiprows=0)
    confounding_data = pd.read_csv(f'{DATA_DIR}/confoundingData.csv', delimiter=DELIMITER, skiprows=0)
    DF_SCREE_PLOT = pd.read_csv(f'{DATA_DIR}/variance_explained.csv', delimiter=DELIMITER, skiprows=0)
    base_df = pd.read_csv(f'{DATA_DIR}/localData.csv', delimiter=DELIMITER, skiprows=0)

    for dir_name in [f.name for f in os.scandir(RESULT_DIR) if f.is_dir()]:
        cluster_nr = int(dir_name.split('_')[1])
        K_VALUES.append(cluster_nr)
        cluster_data = pd.read_csv(f'{RESULT_DIR}/{dir_name}/clustering.csv', delimiter=DELIMITER, skiprows=0)
        df = pd.merge(base_df, cluster_data, on="id")
        df = pd.merge(df, confounding_data, on='id')
        DATAFRAMES_BY_CLUSTER.append(
            {
                'k': cluster_nr,
                'df': df,
            }
        )
        DF_SILHOUETTE.append(
            {
                'k': cluster_nr,
                'df': pd.read_csv(f'{RESULT_DIR}/{dir_name}/silhouette.csv', delimiter=DELIMITER).sort_values(["cluster", "y"], ascending=(True, False)).reset_index(),
            }
        )


@app.callback(
    Output('tabs-content-ct', 'children'),
    Input('tabs-ct', 'value')
)
def render_content(tab):
    if tab == 'tab-confounders':
        return render_confounders()
    elif tab == 'tab-distances':
        return render_distances()
    elif tab == 'tab-clustering-quality':
        return render_clustering_quality()
    elif tab == 'tab-scree-plot':
        return render_scree_plot()
    elif tab == 'tab-scatter-plot':
        return render_scatter_plot()


def render_confounders():
    data_columns = get_data_columns()
    base_content = [
        html.Div(
            [
                html.Span('K', style={'float': 'left', 'width': '15%'}),
                html.Span(
                    dcc.Dropdown(K_VALUES, K_VALUES[0], id='k-confounders', className='fc-dropdown', clearable=False,
                                 style={'float': 'left', 'margin-right': '15%'})),
                html.Span('X axes', style={'float': 'left', 'margin-top': '5px'}),
                html.Span(dcc.Dropdown(data_columns, data_columns[0], id='xaxis-dropdown', className='fc-dropdown',
                                       clearable=False, style={'float': 'left', 'margin-right': '5%'})),
                html.Span('Y axes', style={'float': 'left', 'margin-top': '5px'}),
                html.Span(dcc.Dropdown(data_columns, data_columns[1], id='yaxis-dropdown', className='fc-dropdown',
                                       clearable=False, style={'float': 'left', 'margin-right': '40%'})),
            ]
        ),
        html.Div(get_confounding_factors_filter('confounders'), className='confounding-factors-filter-ct'),
        dcc.Graph(id='confounders-scatter', className='confounders-scatter')
    ]
    return html.Div(base_content)


@app.callback(
    Output('confounders-scatter', 'figure'),
    Input('k-confounders', 'value'),
    Input('xaxis-dropdown', 'value'),
    Input('yaxis-dropdown', 'value'),
    Input({'type': 'filter-checklist-confounders', 'index': ALL}, 'value'),
    Input({'type': 'filter-range-slider-confounders', 'index': ALL}, 'value'),
)
def filter_k_confounders(value, xaxis, yaxis, checklist_values, range_values):
    confounding_df = get_df_by_k_value(value, DATAFRAMES_BY_CLUSTER)
    # filter base dataframe
    index_list = filter_dataframe_on_counfounding_factors(confounding_df, checklist_values, range_values)
    confounding_df = confounding_df[confounding_df.index.isin(index_list)]
    cluster_values_list = confounding_df.cluster.unique()
    nr_of_confounding_factors = len(CONFOUNDING_META.index)
    nr_rows = nr_of_confounding_factors + value
    nr_cols = nr_of_confounding_factors

    specs, subplot_titles = get_specs_for_matrix(nr_rows, nr_cols)
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
            x=df[xaxis],
            y=df[yaxis],
            mode='markers',
            name=f'Cluster {i}',
            marker={
                "size": 10,
                "color": color,
            }
        )
        fig.append_trace(scatter_plot, row=1, col=1)
        path = confidence_ellipse(df[xaxis],
                                  df[yaxis])
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
        for j in range(0, len(CONFOUNDING_META.index)):
            col = CONFOUNDING_META.iloc[j]['name']
            data_type = CONFOUNDING_META.iloc[j]['data_type']
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
    # add log transform buttons
    fig.update_layout(updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0,
            y=1.1,
            buttons=list([
                dict(
                    args=[{'xaxis': {'type': 'scatter'}}],
                    # , 'yaxis': {'type': 'scatter'}, 'y': [df[yaxis].min(), df[yaxis].max()]}],
                    label="Linear",
                    method="relayout"
                ),
                dict(
                    args=[{'xaxis': {'type': 'log'}}],
                    # , 'yaxis': {'type': 'log'}, 'y': [df[yaxis].min(), df[yaxis].max()]}],
                    label="Log",
                    method="relayout"
                )
            ])
        )
    ])
    return fig


def render_distances():
    return html.Div([
        dcc.Graph(id="distance_graph"),
        html.Div(
            children=get_confounding_factors_filter('distance'),
            style={'margin': '0 auto', 'width': '70%'}
        )
    ])


@app.callback(
    Output("distance_graph", "figure"),
    Input({'type': 'filter-checklist-distance', 'index': ALL}, 'value'),
    Input({'type': 'filter-range-slider-distance', 'index': ALL}, 'value'),
)
def filter_heatmap(checklist_values, range_values):
    confounding_df = get_df_by_k_value(K_VALUES[0], DATAFRAMES_BY_CLUSTER)
    index_list = filter_dataframe_on_counfounding_factors(confounding_df, checklist_values, range_values)
    df = DISTANCE_DF[DISTANCE_DF.index.isin(index_list)]
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


def filter_dataframe_on_counfounding_factors(confounding_df, checklist_values, range_values):
    confounding_length = len(CONFOUNDING_META.index)
    # Filter data based on active filters
    checklist_index = range_index = 0
    for j in range(0, confounding_length):
        col = CONFOUNDING_META.iloc[j]["name"]
        data_type = CONFOUNDING_META.iloc[j]['data_type']
        if data_type == 'continuous':
            range_list = range_values[range_index]
            confounding_df = confounding_df.loc[confounding_df[col].between(range_list[0], range_list[1])]
            range_index += 1
        elif data_type == 'discrete':
            checklist = checklist_values[checklist_index]
            confounding_df = confounding_df.loc[confounding_df[col].isin(checklist)]
            checklist_index += 1
    return confounding_df.index.tolist()


def render_clustering_quality():
    return html.Div([
        html.Div(
            [
                html.Span('K', style={'float': 'left', 'margin-top': '6px'}),
                html.Span(dcc.Dropdown(K_VALUES, K_VALUES[0], id='k-labels', className='fc-dropdown', clearable=False,
                                       style={'float': 'left'}))
            ],
            style={'height': '60px', 'width': '100px', 'margin': '20px 70px'}
        ),
        dcc.Graph(id="cluster_quality_graph"),
    ])


@app.callback(
    Output('cluster_quality_graph', 'figure'),
    Input('k-labels', 'value')
)
def filter_k_label(value):
    df = get_df_by_k_value(value, DF_SILHOUETTE)
    avg_value = "{:.2f}".format(df['y'].mean())
    cluster_values_list = df.cluster.unique()

    fig = go.Figure()
    for i in cluster_values_list:
        fig.add_trace(
            go.Bar(
                y=df[df['cluster'] == i]['y'],
                x=df[df['cluster'] == i].index,
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
        x0=0, y0=avg_value, x1=df['x'].max(), y1=avg_value,
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


def render_scree_plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        mode='lines+markers',
        x=DF_SCREE_PLOT['component'],
        y=DF_SCREE_PLOT['eigenvalue'],
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


def render_scatter_plot():
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


def get_specs_for_matrix(rows, cols):
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
            for j in range(0, len(CONFOUNDING_META.index)):
                current_specs_row.append(
                    {'type': 'xy' if CONFOUNDING_META.iloc[j]['data_type'] == 'continuous' else 'pie'})
                subplot_titles.append(CONFOUNDING_META.iloc[j]['name'].capitalize())
        specs.append(current_specs_row)
    return specs, subplot_titles


def get_confounding_factors_filter(id_pre_tag):
    html_elem_list = []
    confounding_df = get_df_by_k_value(K_VALUES[0], DATAFRAMES_BY_CLUSTER)
    confounding_length = len(CONFOUNDING_META.index)
    for j in range(0, confounding_length):
        col = CONFOUNDING_META.iloc[j]["name"]
        data_type = CONFOUNDING_META.iloc[j]['data_type']
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
                                },
                                className="fc-checklist"
                            ),
                            style={'width': '75%', 'float': 'left', 'margin-left': '20px'}
                        ),
                    ],
                    style={'width': '100%', 'display': 'block'}
                )
            )
    return html_elem_list


def get_data_columns():
    # To be changed later to automatic approach
    return ['x', 'y', 'z']


def get_df_by_k_value(k_value, base_obj):
    for k_obj in base_obj:
        if k_obj['k'] == k_value:
            return k_obj['df']
    return []

if __name__ == '__main__':
    assembleDataframes()
    app.run_server(debug=True)
