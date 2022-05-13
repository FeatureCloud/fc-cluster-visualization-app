import os

import dash_bio
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dash_table import DataTable
from dash.dependencies import Input, Output, ALL, State
import plotly.graph_objects as go
import pandas as pd

import numpy as np
from plotly.subplots import make_subplots
from plotly.tools import DEFAULT_PLOTLY_COLORS

DISTANCE_DF = []
CONFOUNDING_META = []
CONFOUNDING_META_BASE = []
DATA_COLUMNS = []
DF_SCREE_PLOT = []
K_VALUES = []
DATAFRAMES_BY_K_VALUE = []
DF_SILHOUETTE = []
DELIMITER = ';'
DATA_DIR = ''
RESULT_DIR = ''
K_VALUE_CONFOUNDERS = 0
K_VALUE_DISTANCE = 0
HEATMAP_INDEX_LIST = []
MAX_NR_OF_CONFOUNDING_FACTORS_TO_DISPLAY = 5

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


def setup(env):
    global DATA_DIR, RESULT_DIR

    if env == 'fc':
        DATA_DIR = '/mnt/input'
    else:
        DATA_DIR = "./data"

    RESULT_DIR = f'{DATA_DIR}/results'


def assemble_dataframes():
    global DISTANCE_DF, CONFOUNDING_META_BASE, CONFOUNDING_META, DATAFRAMES_BY_K_VALUE, DF_SILHOUETTE, DF_SCREE_PLOT, \
        K_VALUES, DATA_COLUMNS
    DATAFRAMES_BY_K_VALUE = []
    try:
        DISTANCE_DF = pd.read_csv(f'{DATA_DIR}/distanceMatrix.csv', delimiter=DELIMITER, skiprows=0, index_col=0)
    except IOError:
        # @TODO display toast message in UI
        print("Warning: distanceMatrix.csv does not appear to exist.")

    try:
        CONFOUNDING_META_BASE = pd.read_csv(f'{DATA_DIR}/confoundingData.meta', delimiter=DELIMITER, skiprows=0)
        confounding_data = pd.read_csv(f'{DATA_DIR}/confoundingData.csv', delimiter=DELIMITER, skiprows=0)
        if len(CONFOUNDING_META) == 0:
            if len(CONFOUNDING_META_BASE) > MAX_NR_OF_CONFOUNDING_FACTORS_TO_DISPLAY:
                CONFOUNDING_META = CONFOUNDING_META_BASE.head(MAX_NR_OF_CONFOUNDING_FACTORS_TO_DISPLAY)
            else:
                CONFOUNDING_META = CONFOUNDING_META_BASE
        confounding_data_expected_column_list = CONFOUNDING_META['name'].tolist()
        confounding_data_expected_column_list.append('id')
        # keep only the selected confounding factors
        confounding_data = confounding_data[
            confounding_data.columns.intersection(confounding_data_expected_column_list)]
    except IOError:
        # @TODO display toast message in UI
        print("Warning: Confounding data is missing")
        confounding_data = []

    try:
        DF_SCREE_PLOT = pd.read_csv(f'{DATA_DIR}/variance_explained.csv', delimiter=DELIMITER, skiprows=0)
    except IOError:
        # @TODO display toast message in UI
        print("Warning: variance_explained.csv does not exist")

    base_df = pd.read_csv(f'{DATA_DIR}/localData.csv', delimiter=DELIMITER, skiprows=0)
    DATA_COLUMNS = base_df.columns.to_list()
    DATA_COLUMNS.remove('id')
    if 'client_id' in DATA_COLUMNS:
        DATA_COLUMNS.remove('client_id')

    for dir_name in [f.name for f in os.scandir(RESULT_DIR) if f.is_dir()]:
        cluster_nr = int(dir_name.split('_')[1])
        K_VALUES.append(cluster_nr)
        cluster_data = pd.read_csv(f'{RESULT_DIR}/{dir_name}/clustering.csv', delimiter=DELIMITER, skiprows=0)
        df = pd.merge(base_df, cluster_data, on="id")

        # put cluster column in different place to be used in hovertemplate, based on customdata parameter
        column_list = df.columns.to_list()
        column_list.remove('cluster')
        if 'client_id' in column_list:
            index = column_list.index('client_id') + 1
        else:
            index = column_list.index('id') + 1
        column_list.insert(index, 'cluster')
        df = df[column_list]
        if len(confounding_data) > 0:
            df = pd.merge(df, confounding_data, on='id')
        DATAFRAMES_BY_K_VALUE.append(
            {
                'k': cluster_nr,
                'df': df,
            }
        )
        DF_SILHOUETTE.append(
            {
                'k': cluster_nr,
                'df': pd.read_csv(f'{RESULT_DIR}/{dir_name}/silhouette.csv', delimiter=DELIMITER).sort_values(
                    ["cluster", "y"], ascending=(True, False)).reset_index(),
            }
        )


def create_dash(path_prefix):
    app = Dash(__name__,
               requests_pathname_prefix=path_prefix,
               title='FeatureCloud Cluster Visualization App')
    distance_style = {'display': 'none'} if len(DISTANCE_DF) == 0 else {}
    scree_plot_style = {'display': 'none'} if len(DF_SCREE_PLOT) == 0 else {}

    app.layout = html.Div([
        html.H2('FeatureCloud Cluster Visualization App', className='fc-header'),
        dcc.Tabs(id="tabs-ct", value='tab-confounders', children=[
            dcc.Tab(label='Confounders', value='tab-confounders'),
            dcc.Tab(label='Distances', value='tab-distances', style=distance_style),
            dcc.Tab(label='Clustering Quality', value='tab-clustering-quality'),
            dcc.Tab(label='Scree plot', value='tab-scree-plot', style=scree_plot_style),
            dcc.Tab(label='Help', value='tab-help'),
        ]),
        html.Div(id='tabs-content-ct', style={'width': '75%', 'margin': '0 auto'}),
    ])

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
        elif tab == 'tab-help':
            return render_help()

    def render_confounders():
        confounding_df = get_df_by_k_value(K_VALUES[0], DATAFRAMES_BY_K_VALUE)
        datatable_columns = confounding_df.columns.to_list()
        height_multiplier = 0 if len(CONFOUNDING_META) == 0 else len(CONFOUNDING_META.index)
        confounders_filter_height = f'{32 + height_multiplier * 40}px'
        id_post_tag = 'confounders-tab'
        cluster_client_switch_style = {}
        if 'client_id' not in confounding_df:
            cluster_client_switch_style['display'] = 'none'
        base_content = [
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                children=[
                                    html.Label('Client', id='client-label'),
                                    html.Span(
                                        daq.BooleanSwitch(
                                            id='cluster-client-switch',
                                            on=True,
                                            label='Cluster',
                                            labelPosition='right',
                                        ),
                                        style={'float': 'left'}
                                    ),
                                ],
                                style=cluster_client_switch_style
                            ),
                            dbc.Col(children=get_k_filter(id_post_tag)),
                            dbc.Col(get_cluster_values_filter(id_post_tag)),
                            dbc.Col(
                                children=
                                [
                                    html.Span('X axes', style={'float': 'left', 'margin-top': '5px'}),
                                    html.Span(dcc.Dropdown(DATA_COLUMNS, DATA_COLUMNS[0], id='xaxis-dropdown',
                                                           className='fc-dropdown', clearable=False,
                                                           style={'float': 'left'})),
                                    html.Span('Y axes',
                                              style={'float': 'left', 'margin-top': '5px', 'margin-left': '10px'}),
                                    html.Span(dcc.Dropdown(DATA_COLUMNS, DATA_COLUMNS[1], id='yaxis-dropdown',
                                                           className='fc-dropdown', clearable=False,
                                                           style={'float': 'left'})),
                                ]
                            ),
                            dbc.Col(
                                html.Span(
                                    daq.BooleanSwitch(
                                        id='use-pie-charts-switch',
                                        on=True,
                                        label='Use pie chart for discrete data type',
                                        labelPosition='right',
                                    ),
                                    style={'float': 'left'}
                                ),
                            ),
                        ],
                        style={'margin-top': '20px'}
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        get_confounding_factors_filter('confounders'),
                        id='confounding-factors-filter-ct',
                        className='confounding-factors-filter-ct',
                        style={'height': confounders_filter_height}
                    ),
                )
            ),
            dbc.Row(
                dcc.Graph(id='confounders-scatter', className='confounders-scatter'),
            ),
            dbc.Row(
                dbc.Fade(
                    html.Div([
                        html.Div(children=[
                            html.Span(
                                daq.BooleanSwitch(
                                    id='view-as-diagram-switch',
                                    on=False,
                                    label='View selected data in diagrams',
                                    labelPosition='right',
                                ),
                                style={'float': 'left'}
                            ),
                            html.Span(
                                '.csv',
                                style={'float': 'right', 'margin-top': '7px'}
                            ),
                            html.Span(
                                dbc.Input(id='selection-group-name', placeholder="Outlier_group"),
                                style={'float': 'right', 'margin-left': '10px'}
                            ),
                            html.Span(
                                'Filename: ',
                                style={'float': 'right', 'margin-top': '7px'}
                            ),
                            dbc.Button('Download', id='btn-download', color='secondary', className='me-1',
                                       style={'float': 'right', 'margin-left': '10px'}),
                            dcc.Checklist(
                                ['Download inverse selection'], [], inline=True,
                                id='download-inverse-selection', className="fc-checklist",
                                style={'float': 'right', 'margin-left': '10px'}
                            ),
                        ], style={'height': '40px'}),
                        dbc.Collapse(
                            DataTable(
                                id='selection-datatable',
                                columns=[{
                                    'name': col_name.capitalize(),
                                    'id': col_name,
                                } for col_name in datatable_columns],
                            ),
                            id='collapse-datatable',
                            is_open=True
                        ),
                        dbc.Collapse(
                            dcc.Graph(id='selection-graph', className='confounders-scatter'),
                            id='collapse-selection-graph',
                            is_open=False
                        ),
                        dcc.Download(id="download-dataframe-csv"),
                    ]),
                    id='fade-ct',
                    is_in=False,
                    appear=False
                ),
            )
        ]
        return html.Div(base_content)

    @app.callback(
        Output('confounders-scatter', 'figure'),
        Output('cluster-values-checklist-confounders-tab', 'options'),
        Output('cluster-values-checklist-confounders-tab', 'value'),
        Input('k-filter-confounders-tab', 'value'),
        Input('cluster-values-checklist-confounders-tab', 'value'),
        Input('xaxis-dropdown', 'value'),
        Input('yaxis-dropdown', 'value'),
        Input({'type': 'filter-checklist-confounders', 'index': ALL}, 'value'),
        Input({'type': 'filter-range-slider-confounders', 'index': ALL}, 'value'),
        Input('use-pie-charts-switch', 'on'),
        Input('cluster-client-switch', 'on'),
    )
    def filter_confounders_view(k_value, selected_clusters, xaxis, yaxis, checklist_values, range_values,
                                use_pie_charts, use_clusters):
        global K_VALUE_CONFOUNDERS
        if use_clusters:
            clustering_field = 'cluster'
        else:
            clustering_field = 'client_id'
        confounding_df = get_df_by_k_value(k_value, DATAFRAMES_BY_K_VALUE)
        cluster_checklist_values = get_cluster_values_list(confounding_df)
        # Detect if K value has changed, to reset checklist values to all values selected
        if K_VALUE_CONFOUNDERS != k_value:
            selected_clusters = cluster_checklist_values
        K_VALUE_CONFOUNDERS = k_value
        # filter base dataframe
        index_list = filter_dataframe_on_confounding_factors(confounding_df, selected_clusters, checklist_values,
                                                              range_values, use_clusters)
        confounding_df = confounding_df[confounding_df.index.isin(index_list)]
        fig = get_figure_with_subplots(confounding_df, k_value, xaxis, yaxis, use_pie_charts, clustering_field)
        return fig, cluster_checklist_values, selected_clusters

    def get_figure_with_subplots(confounding_df, k_value, xaxis, yaxis, use_pie_charts, clustering_field):
        cluster_values_list = confounding_df[clustering_field].unique()
        nr_of_confounding_factors = 0 if len(CONFOUNDING_META) == 0 else len(CONFOUNDING_META.index)

        if 'client_id' in confounding_df:
            client_id_present = True
        else:
            client_id_present = False

        nr_cols = 1 if nr_of_confounding_factors == 0 else nr_of_confounding_factors
        if nr_of_confounding_factors == 0:
            nr_rows = 1
            nr_cols = 1
        else:
            if client_id_present:
                nr_rows = nr_of_confounding_factors + len(cluster_values_list) + 1
            else:
                nr_rows = nr_of_confounding_factors + k_value + 1

        specs, subplot_titles = get_specs_for_matrix(nr_rows, nr_cols, use_pie_charts, clustering_field)
        fig = make_subplots(
            rows=nr_rows,
            cols=nr_cols,
            specs=specs,
            subplot_titles=subplot_titles,
        )
        for i in cluster_values_list:
            color = DEFAULT_PLOTLY_COLORS[i]
            df = confounding_df[confounding_df[clustering_field] == i]
            marker = {
                "size": 10,
                "color": color,
            }
            
            # construct customdata for hover info
            customdata = []
            for index, row in df.iterrows():
                if client_id_present is True:
                    customdata.append([row['id'], row['cluster'], row['client_id']])
                else:
                    customdata.append([row['id'], row['cluster']])

            if client_id_present:
                if clustering_field == 'cluster':
                    marker['symbol'] = df['client_id']
                    hovertemplate = "Sample: %{customdata[0]}<br>Client id: %{customdata[1]}"
                else:
                    marker['symbol'] = df['cluster']
                    hovertemplate = "Sample: %{customdata[0]}<br>Cluster: %{customdata[2]}"
            else:
                hovertemplate = "Sample: %{customdata[0]}"

            scatter_plot = go.Scatter(
                x=df[xaxis],
                y=df[yaxis],
                mode='markers',
                name=f'{clustering_field.capitalize()} {i}',
                marker=marker,
                customdata=df,
                hovertemplate=hovertemplate,
                legendgroup="0",
                legendgrouptitle=dict(text=clustering_field.capitalize()),
                showlegend=True,
            )
            fig.append_trace(scatter_plot, row=1, col=1)
            path = confidence_ellipse(df[xaxis], df[yaxis])
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
            df = confounding_df[confounding_df[clustering_field] == i]
            for j in range(0, nr_of_confounding_factors):
                col = CONFOUNDING_META.iloc[j]['name']
                data_type = CONFOUNDING_META.iloc[j]['data_type']
                if data_type == 'continuous' or data_type == 'ordinal' or not use_pie_charts:
                    # add histogram
                    bar_continuous = go.Histogram(
                        x=df[col],
                        marker={'color': color},
                        hovertemplate=col.capitalize() + ' group: %{x}<br>Count: %{y}',
                        showlegend=False,
                    )
                    fig.add_trace(bar_continuous, row=i + nr_cols, col=j + 1)
                elif data_type == 'discrete' and use_pie_charts:
                    # add pie chart
                    pie_values_list = []
                    discrete_val_list = confounding_df[col].unique()

                    for discrete_val in discrete_val_list:
                        pie_values_list.append(df[df[col] == discrete_val].count()[col])
                    pie_chart = go.Pie(
                        labels=discrete_val_list,
                        values=pie_values_list,
                        showlegend=True,
                        legendgroup=str(j),
                        legendgrouptitle=dict(text=col.capitalize())
                    )
                    fig.add_trace(pie_chart, row=i + nr_cols, col=j + 1)

        # Add summary row for confounding factors
        for j in range(0, nr_of_confounding_factors):
            col = CONFOUNDING_META.iloc[j]['name']
            data_type = CONFOUNDING_META.iloc[j]['data_type']
            for i in cluster_values_list:
                df = confounding_df[confounding_df[clustering_field] == i]
                color = DEFAULT_PLOTLY_COLORS[i]
                # add histogram
                bar_continuous = go.Histogram(
                    x=df[col],
                    marker={'color': color},
                    hovertemplate=f'{clustering_field.capitalize()} {i}<br>' + col.capitalize() + ' group: %{x}<br>Count: %{y}',
                    showlegend=False,
                )
                fig.add_trace(bar_continuous, row=nr_rows, col=j + 1)

        # add log transform buttons
        fig.update_layout(
            clickmode='event+select',
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0,
                    y=1.1,
                    xanchor='left',
                    yanchor='top',
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
            ]
        )
        return fig

    @app.callback(
        Output("selection-datatable", "data"),
        Output("fade-ct", "is_in"),
        Output('selection-graph', 'figure'),
        Input('confounders-scatter', 'selectedData'),
        Input('k-filter-confounders-tab', 'value'),
        Input('xaxis-dropdown', 'value'),
        Input('yaxis-dropdown', 'value'),
        State('use-pie-charts-switch', 'on'),
    )
    def display_selected(selected_data, k_value, xaxis, yaxis, use_pie_charts):
        if selected_data is None:
            return [], False, go.Figure()
        confounding_df = get_df_by_k_value(k_value, DATAFRAMES_BY_K_VALUE)
        datatable_columns = confounding_df.columns.to_list()
        records = []
        selected_points = selected_data['points']
        for point in selected_points:
            records.append(point['customdata'])

        df = pd.DataFrame(data=records, columns=datatable_columns)
        data_ob = df.to_dict('records')
        fig = get_figure_with_subplots(df, k_value, xaxis, yaxis, use_pie_charts, 'cluster')
        return data_ob, True, fig

    @app.callback(
        Output('download-dataframe-csv', 'data'),
        Input('btn-download', 'n_clicks'),
        State('k-filter-confounders-tab', 'value'),
        State('download-inverse-selection', 'value'),
        State("selection-datatable", "data"),
        State("selection-datatable", "column"),
        State('selection-group-name', 'value')
    )
    def download_selected(n_clicks, k_value, inverse_selection, data, columns, group_name):
        if data is None or len(data) == 0:
            return

        default_file_name = 'Outlier_Group'
        df = pd.DataFrame(data=data, columns=columns)
        if inverse_selection == ['Download inverse selection']:
            df = pd.DataFrame(data=filter_dataframe_inverse_on_id(k_value, df['id'].tolist()), columns=columns)

        if group_name is None:
            group_name = default_file_name
        else:
            group_name = group_name.strip()
            if len(group_name) == 0:
                group_name = default_file_name

        return dcc.send_data_frame(df.to_csv, f'{group_name}.csv')

    @app.callback(
        Output('collapse-datatable', 'is_open'),
        Output('collapse-selection-graph', 'is_open'),
        Input('view-as-diagram-switch', 'on')
    )
    def switch_datatable_view(on):
        return not on, on

    @app.callback(
        Output("distance_graph", "figure"),
        Output('cluster-values-checklist-distances-tab', 'options'),
        Output('cluster-values-checklist-distances-tab', 'value'),
        Output('dataframe-empty-toast', 'is_open'),
        Input('k-filter-distances-tab', 'value'),
        Input('cluster-values-checklist-distances-tab', 'value'),
        Input({'type': 'filter-checklist-distance', 'index': ALL}, 'value'),
        Input({'type': 'filter-range-slider-distance', 'index': ALL}, 'value'),
    )
    def filter_heatmap(k_value, selected_clusters, checklist_values, range_values):
        global K_VALUE_DISTANCE, HEATMAP_INDEX_LIST
        confounding_df = get_df_by_k_value(k_value, DATAFRAMES_BY_K_VALUE)
        cluster_checklist_values = get_cluster_values_list(confounding_df)
        # Detect if K value has changed, to reset checklist values to all values selected
        if K_VALUE_DISTANCE != k_value:
            selected_clusters = cluster_checklist_values
        K_VALUE_DISTANCE = k_value
        index_list = filter_dataframe_on_confounding_factors(confounding_df, selected_clusters,
                                                              checklist_values, range_values, True)
        display_error_toaster = False
        if len(index_list) == 0:
            display_error_toaster = True
            index_list = HEATMAP_INDEX_LIST
        else:
            HEATMAP_INDEX_LIST = index_list

        df = DISTANCE_DF[DISTANCE_DF.index.isin(index_list)]
        return dash_bio.Clustergram(
            data=df,
            column_labels=list(df.columns.values),
            row_labels=list(df.index),
            height=800,
            width=1400,
            # display_ratio=[0.1, 0.7],
            hidden_labels='rows, columns'
        ), cluster_checklist_values, selected_clusters, display_error_toaster

    @app.callback(
        Output('cluster_quality_graph', 'figure'),
        Input('k-labels', 'value')
    )
    def filter_k_label(value):
        df = get_df_by_k_value(value, DF_SILHOUETTE)
        avg_value = "{:.2f}".format(df['y'].mean())
        cluster_values_list = df.cluster.unique()
        fig = go.Figure()
        for i in reversed(cluster_values_list):
            color = DEFAULT_PLOTLY_COLORS[i]
            fig.add_trace(
                go.Bar(
                    x=df[df['cluster'] == i]['y'],
                    y=df[df['cluster'] == i].index,
                    orientation='h',
                    name=f'Cluster {i}',
                    marker={
                        "line": {
                            "width": 0,
                        },
                        "color": color
                    }
                )
            )
        # Add avg line on top
        fig.add_shape(
            type="line",
            y0=0, x0=avg_value, y1=df['x'].max(), x1=avg_value,
            line=dict(
                color="Red",
                width=2,
                dash="dashdot",
            )
        )
        fig.update_layout(
            title=f'Clusters silhouette plot<br>Average silhouette width: {str(avg_value)}',
            yaxis={
                "title": "",
                "showticklabels": False,
            },
            xaxis={"title": "Silhouette width Si"},
            bargap=0.0,
            showlegend=True,
            legend={
                "title": "Clusters",
            },
        )
        return fig

    @app.callback(
        Output('confounding-modal', 'is_open'),
        Input('btn-open-confounding-modal', 'n_clicks'),
        Input('btn-set-confounding-factors', 'n_clicks'),
        State('confounding-modal', 'is_open')
    )
    def toggle_modal(n_open, n_close, is_open):
        if n_open or n_close:
            return not is_open
        return is_open

    return app


def render_distances():
    id_post_tag = 'distances-tab'
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(children=get_k_filter(id_post_tag)),
                    dbc.Col(get_cluster_values_filter(id_post_tag)),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        children=get_confounding_factors_filter('distance'),
                        style={'margin-top': '20px'}
                    )
                )
            ),
            dbc.Row(
                dbc.Col(dcc.Graph(id="distance_graph"))
            ),
            dbc.Toast(
                [html.P("Dataframe is empty. Clustergram cannot be calculated.", className="mb-0")],
                id="dataframe-empty-toast",
                header="Error",
                duration=4000,
                is_open=False,
                icon="danger",
                style={"position": "fixed", "top": 66, "right": 10, "width": 350},
            ),
        ],
        style={'margin-top': '20px'}
    )


def filter_dataframe_on_confounding_factors(confounding_df, selected_clusters, checklist_values, range_values, use_clusters):
    selected_cluster_ids = []
    if len(CONFOUNDING_META) == 0:
        return confounding_df.index.tolist()
    if len(selected_clusters) > 0:
        for cluster_value in selected_clusters:
            cluster_id = int(cluster_value.split()[1])
            selected_cluster_ids.append(cluster_id)
    if use_clusters:
        confounding_df = confounding_df.loc[confounding_df['cluster'].isin(selected_cluster_ids)]

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
        elif data_type == 'discrete' or data_type == 'ordinal':
            checklist = checklist_values[checklist_index]
            confounding_df = confounding_df.loc[confounding_df[col].isin(checklist)]
            checklist_index += 1
    return confounding_df.index.tolist()


def filter_dataframe_inverse_on_id(k_value, selected_ids):
    confounding_df = get_df_by_k_value(k_value, DATAFRAMES_BY_K_VALUE)
    selected_data = confounding_df.loc[~confounding_df['id'].isin(selected_ids)]
    return selected_data


def render_clustering_quality():
    return html.Div([
        dbc.Row([
            dbc.Col(children=
            [
                html.Span('K', style={'float': 'left', 'margin-top': '6px'}),
                html.Span(dcc.Dropdown(K_VALUES, K_VALUES[0], id='k-labels', className='fc-dropdown', clearable=False,
                                       style={'float': 'left'}))
            ],
                style={'height': '60px', 'width': '100px', 'margin': '20px 70px'}
            ),
        ]),
        dbc.Row([
            dcc.Graph(id="cluster_quality_graph"),
        ])
    ])


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


def render_help():
    f = open('README.md', 'r')
    return dcc.Markdown(f.read())


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


def get_specs_for_matrix(rows, cols, use_pie_charts, clustering_field):
    specs = []
    subplot_titles = []
    nr_of_confounding_factors = 0 if len(CONFOUNDING_META) == 0 else len(CONFOUNDING_META.index)
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
            for j in range(0, nr_of_confounding_factors):
                title = ''
                if rows != i:
                    current_specs_row.append(
                        {'type': 'pie' if CONFOUNDING_META.iloc[j][
                                              'data_type'] == 'discrete' and use_pie_charts else 'xy'})
                    title = f'{clustering_field.capitalize()} {i - cols}: {CONFOUNDING_META.iloc[j]["name"].capitalize()}'
                else:
                    current_specs_row.append({'type': 'xy'})
                    title = f'All {clustering_field}s: {CONFOUNDING_META.iloc[j]["name"].capitalize()}'
                subplot_titles.append(title)
        specs.append(current_specs_row)
    return specs, subplot_titles


def get_confounding_factors_filter(id_pre_tag):
    html_elem_list = []
    if len(CONFOUNDING_META) == 0:
        return html_elem_list
    confounding_df = get_df_by_k_value(K_VALUES[0], DATAFRAMES_BY_K_VALUE)
    confounding_length = len(CONFOUNDING_META.index)
    confounding_base_length = len(CONFOUNDING_META_BASE)
    confounding_selector_options = []
    for j in range(0, confounding_base_length):
        confounding_selector_options.append(CONFOUNDING_META_BASE.iloc[j]['name'].capitalize())

    html_elem_list.append(
        dbc.Row(
            children=
            [
                dbc.Col(html.H5("Confounding factors filter")),
            ]
        )
    )
    for j in range(0, confounding_length):
        col = CONFOUNDING_META.iloc[j]["name"]
        data_type = CONFOUNDING_META.iloc[j]['data_type']
        if data_type == 'continuous':
            # add range slider
            col_min = confounding_df[col].min()
            col_max = confounding_df[col].max()
            html_elem_list.append(
                dbc.Row(
                    dbc.Col(
                        children=
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
                        ]
                    )
                )
            )
        elif data_type == 'discrete' or data_type == 'ordinal':
            # add checklist
            discrete_val_list = confounding_df[col].unique()
            html_elem_list.append(
                dbc.Row(
                    dbc.Col(
                        children=
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
                        ]
                    )
                )
            )
    return html_elem_list


def get_cluster_values_list(df):
    cluster_values = df.cluster.unique()
    cluster_values_list = []
    for i in cluster_values:
        cluster_values_list.append(f'Cluster {i}')

    return cluster_values_list


def get_client_values_list(df):
    client_values = df.client_id.unique()
    client_values_list = []
    for i in client_values:
        client_values_list.append(f'Client {i}')

    return client_values_list


def get_df_by_k_value(k_value, base_obj):
    for k_obj in base_obj:
        if k_obj['k'] == k_value:
            return k_obj['df']
    return []


def get_k_filter(id_post_tag):
    return [

        html.Span('K', style={'float': 'left', 'margin-top': '5px'}),
        html.Span(
            dcc.Dropdown(K_VALUES, K_VALUES[0], id=f'k-filter-{id_post_tag}', className='fc-dropdown',
                         clearable=False, style={'float': 'left', 'margin-right': '15%'})),
    ]


def get_cluster_values_filter(id_post_tag):
    confounding_df = get_df_by_k_value(K_VALUES[0], DATAFRAMES_BY_K_VALUE)
    cluster_values_list = get_cluster_values_list(confounding_df)
    return html.Span(
        dcc.Checklist(cluster_values_list, cluster_values_list,
                      inline=True, id=f'cluster-values-checklist-{id_post_tag}', className="fc-checklist"),
    )


def get_client_values_filter(id_post_tag):
    confounding_df = get_df_by_k_value(K_VALUES[0], DATAFRAMES_BY_K_VALUE)
    cluster_values_list = get_client_values_list(confounding_df)
    return html.Span(
        dcc.Checklist(cluster_values_list, cluster_values_list,
                      inline=True, id=f'client-values-checklist-{id_post_tag}', className="fc-checklist"),
    )


def vet_source_files(file, requirements):
    return True


def start(env, path_prefix):
    setup(env)
    assemble_dataframes()
    dash = create_dash(path_prefix)

    if env == 'fc':
        dash.run_server(debug=False, port=8050)
    else:
        dash.run_server(debug=True, port=8050)
