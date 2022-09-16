import math
import os
import shutil

import dash_bio
import yaml
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dash_table import DataTable
from dash.dcc import Download
from dash.dependencies import Input, Output, ALL, State
import plotly.graph_objects as go
import pandas as pd

import numpy as np
from flask import request
from plotly.subplots import make_subplots
from plotly.tools import DEFAULT_PLOTLY_COLORS


class fcvisualization:

    def __init__(self):
        self.callback_fn_terminal_state = None
        self.k_value_distance = 0
        self.distance_df = []
        self.confounding_meta = []
        self.data_columns = []
        self.scree_plot = []
        self.k_values = []
        self.dataframes_by_k_value = []
        self.silhouette = []
        self.delimiter = ''
        self.base_dir_fc_env = '/mnt/input'
        self.data_dir = ''
        self.output_dir = ''
        self.k_value_confounders = 0
        self.heatmap_index_list = []
        self.max_nr_of_confounding_factors_to_display = 5
        self.data_errors = ''
        self.volcano_df = []

        # Configurable paths for data files
        self.local_data_path = ''
        self.confounding_data_path = ''
        self.confounding_meta_path = ''
        self.distance_matrix_path = ''
        self.variance_explained_path = ''
        self.k_values_clustering_result_dir = ''
        self.k_values_clustering_file_name = ''
        self.k_values_silhouette_file_name = ''
        self.volcano_data_path = ''
        self.download_dir = ''
        self.env = ''

    def start(self, env, path_prefix, callback_fn):
        def run_fc():
            dash.run_server(debug=False, port=8050)

        def run_native():
            dash.run_server(debug=True, port=8050)

        self.callback_fn_terminal_state = callback_fn
        self.setup(env)
        self.assemble_dataframes()
        dash = self.create_dash(path_prefix)

        if env == 'fc':
            dash.run_server(debug=False, port=8050)
            # process = multiprocessing.Process(target=run_fc)
            # process.start()
        else:
            dash.run_server(debug=True, port=8050)
            # process = multiprocessing.Process(target=run_native)
            # process.start()

    def setup(self, env):
        self.env = env
        self.data_dir = "./data"
        self.output_dir = f'{self.data_dir}/output'

        if self.env == 'fc':
            self.data_dir = '/mnt/input'
            self.output_dir = '/mnt/output'

        self.delimiter = ';'
        self.local_data_path = f'{self.data_dir}/localData.csv'
        self.distance_matrix_path = f'{self.data_dir}/distanceMatrix.csv'
        self.confounding_meta_path = f'{self.data_dir}/confoundingData.meta'
        self.confounding_data_path = f'{self.data_dir}/confoundingData.csv'
        self.variance_explained_path = f'{self.data_dir}/variance_explained.csv'
        self.k_values_clustering_result_dir = f'{self.data_dir}/results'
        self.k_values_clustering_file_name = 'clustering.csv'
        self.k_values_silhouette_file_name = 'silhouette.csv'
        self.volcano_data_path = f'{self.data_dir}/volcano_data.csv'
        self.download_dir = f'{self.output_dir}/downloads'

        if self.env == 'fc':
            # copy input folder content to output folder
            shutil.copytree(self.data_dir, self.output_dir, dirs_exist_ok=True)

            os.chdir('./app')

        # process config.yml if there is any
        config_file_path = self.data_dir + '/config.yml'
        try:
            with open(config_file_path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                if 'fc-cluster-visualization-app' in config:
                    config = config['fc-cluster-visualization-app']
                    if 'self.delimiter' in config:
                        self.delimiter = config['self.delimiter']
                    if 'data-dir' in config:
                        if self.env == 'fc':
                            self.data_dir = os.path.join(self.base_dir_fc_env, config['data-dir'])
                        else:
                            self.data_dir = config['data-dir']
                    if 'local-data-path' in config:
                        if self.env == 'fc':
                            self.local_data_path = os.path.join(self.base_dir_fc_env, config['local-data-path'])
                        else:
                            self.local_data_path = config['local-data-path']
                    if 'distance-matrix-path' in config:
                        if self.env == 'fc':
                            self.distance_matrix_path = os.path.join(self.base_dir_fc_env,
                                                                     config['distance-matrix-path'])
                        else:
                            self.distance_matrix_path = config['distance-matrix-path']
                    if 'confounding-meta-path' in config:
                        if self.env == 'fc':
                            self.confounding_meta_path = os.path.join(self.base_dir_fc_env,
                                                                      config['confounding-meta-path'])
                        else:
                            self.confounding_meta_path = config['confounding-meta-path']
                    if 'confounding-data-path' in config:
                        if self.env == 'fc':
                            self.confounding_data_path = os.path.join(self.base_dir_fc_env,
                                                                      config['confounding-data-path'])
                        else:
                            self.confounding_data_path = config['confounding-data-path']
                    if 'variance-explained-path' in config:
                        if self.env == 'fc':
                            self.variance_explained_path = os.path.join(self.base_dir_fc_env,
                                                                        config['variance-explained-path'])
                        else:
                            self.variance_explained_path = config['variance-explained-path']
                    if 'k-values-clustering-result-dir' in config:
                        if self.env == 'fc':
                            self.k_values_clustering_result_dir = os.path.join(self.base_dir_fc_env,
                                                                               config['k-values-clustering-result-dir'])
                        else:
                            self.k_values_clustering_result_dir = config['k-values-clustering-result-dir']
                    if 'k-values-clustering-file-name' in config:
                        self.k_values_clustering_file_name = config['k-values-clustering-file-name']
                    if 'k-values-silhouette-file-name' in config:
                        self.k_values_silhouette_file_name = config['k-values-silhouette-file-name']
                    if 'volcano-data-path' in config:
                        if self.env == 'fc':
                            self.volcano_data_path = os.path.join(self.base_dir_fc_env,
                                                             config['volcano-data-path'])
                        else:
                            self.volcano_data_path = config['volcano-data-path']
                    if 'download-dir' in config:
                        if self.env == 'fc':
                            self.download_dir = os.path.join(self.output_dir, config['download-dir'])
                        else:
                            self.download_dir = config['download-dir']
        except IOError:
            print('No config file found, will work with default values.')

        if not os.path.isdir(self.download_dir):
            os.makedirs(self.download_dir, exist_ok=True)

        print("Working with the following configuration:")
        print(f'delimiter={self.delimiter}')
        print(f'data-dir={self.data_dir}')
        print(f'local_data_path={self.local_data_path}')
        print(f'distance_matrix_path={self.distance_matrix_path}')
        print(f'confounding_meta_path={self.confounding_meta_path}')
        print(f'confounding_data_path={self.confounding_data_path}')
        print(f'variance_explained_path={self.variance_explained_path}')
        print(f'k_values_clustering_result_dir={self.k_values_clustering_result_dir}')
        print(f'k_values_clustering_file_name={self.k_values_clustering_file_name}')
        print(f'k_values_silhouette_file_name={self.k_values_silhouette_file_name}')
        print(f'volcano_data_path={self.volcano_data_path}')
        print(f'download_dir={self.download_dir}')
        print(f'Current directory is: {os.getcwd()}')

    def assemble_dataframes(self):
        self.dataframes_by_k_value = []
        if not os.path.isdir(self.data_dir):
            self.data_errors += "Data folder is missing."

        local_data_present = True
        try:
            base_df = pd.read_csv(self.local_data_path, delimiter=self.delimiter, skiprows=0)
            nr_of_samples = len(base_df.index)
        except IOError:
            print(f'Did not find local data file in: {self.local_data_path}')
            local_data_present = False

        try:
            self.volcano_df = pd.read_csv(self.volcano_data_path, delimiter=self.delimiter, skiprows=0)
            if 'EFFECTSIZE' not in self.volcano_df.columns or 'P' not in self.volcano_df.columns or 'SNP' not in self.volcano_df.columns \
                    or 'GENE' not in self.volcano_df.columns:
                self.data_errors += f'Error: Wrong self.delimiter ({self.delimiter}) or missing column(s) in data set for volcano plot. ' \
                                    f'Required columns are: "EFFECTSIZE", "P", "SNP", "GENE".\n'
        except IOError:
            self.data_errors += f'Warning: {self.volcano_data_path} does not exist.\n'
        except pd.errors.EmptyDataError:
            self.data_errors += f'Error: {self.volcano_data_path} is empty.\n'

        if not local_data_present:
            if len(self.volcano_df) == 0:
                self.data_errors += "Local data is missing"
            return

        try:
            self.distance_df = pd.read_csv(self.distance_matrix_path, delimiter=self.delimiter, skiprows=0, index_col=0)
            if len(self.distance_df.columns) != len(self.distance_df.index) or len(
                    self.distance_df.index) != nr_of_samples:
                self.data_errors += f'Data inconsistency in {self.distance_matrix_path}. Number of samples are not matching \n'
        except IOError:
            self.data_errors += f'Warning: {self.distance_matrix_path} does not appear to exist.\n'
        except pd.errors.EmptyDataError:
            self.data_errors += "Error: Distance matrix is empty.\n"

        try:
            self.confounding_meta = pd.read_csv(self.confounding_meta_path, delimiter=self.delimiter, skiprows=0)
            confounding_data = pd.read_csv(self.confounding_data_path, delimiter=self.delimiter, skiprows=0)
            if len(self.confounding_meta) > self.max_nr_of_confounding_factors_to_display:
                self.confounding_meta = self.confounding_meta.head(self.max_nr_of_confounding_factors_to_display)
                self.data_errors += f'Warning: The application supports a maximum of {self.max_nr_of_confounding_factors_to_display} of confounding factors. ' \
                                    f'The first {self.max_nr_of_confounding_factors_to_display} will be displayed \n'
            confounding_data_expected_column_list = self.confounding_meta['name'].tolist()
            confounding_data_expected_column_list.append('id')
            # keep only the selected confounding factors
            confounding_data = confounding_data[
                confounding_data.columns.intersection(confounding_data_expected_column_list)]
        except IOError:
            self.data_errors += "Warning: Confounding data is missing.\n"
            confounding_data = []
        except pd.errors.EmptyDataError:
            self.data_errors += "Error: Confounding data and/or meta is empty.\n"
            confounding_data = []

        try:
            self.scree_plot = pd.read_csv(self.variance_explained_path, delimiter=self.delimiter, skiprows=0)
        except IOError:
            self.data_errors += f'Warning: {self.variance_explained_path} does not exist.\n'
        except pd.errors.EmptyDataError:
            self.data_errors += f'Error: {self.variance_explained_path} is empty.\n'

        self.data_columns = base_df.columns.to_list()
        self.data_columns.remove('id')
        if 'client_id' in self.data_columns:
            self.data_columns.remove('client_id')

        if os.path.isdir(self.data_dir) and os.path.isdir(self.k_values_clustering_result_dir):
            for dir_name in [f.name for f in os.scandir(self.k_values_clustering_result_dir) if f.is_dir()]:
                cluster_nr = int(dir_name.split('_')[1])
                self.k_values.append(cluster_nr)
                cluster_data = pd.read_csv(
                    f'{self.k_values_clustering_result_dir}/{dir_name}/{self.k_values_clustering_file_name}',
                    delimiter=self.delimiter, skiprows=0)
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
                self.dataframes_by_k_value.append(
                    {
                        'k': cluster_nr,
                        'df': df,
                    }
                )
                self.silhouette.append(
                    {
                        'k': cluster_nr,
                        'df': pd.read_csv(
                            f'{self.k_values_clustering_result_dir}/{dir_name}/{self.k_values_silhouette_file_name}',
                            delimiter=self.delimiter).sort_values(
                            ["cluster", "y"], ascending=(True, False)).reset_index(),
                    }
                )
        if len(self.k_values) == 0 or len(self.dataframes_by_k_value) == 0 or len(self.silhouette) == 0:
            self.k_values.append(0)
            base_df['cluster'] = 0
            self.dataframes_by_k_value.append(
                {
                    'k': 0,
                    'df': pd.merge(base_df, confounding_data, on='id') if len(confounding_data) > 0 else base_df
                }
            )
            self.data_errors += "Error: Clustering information is missing or corrupt.\n"

    def create_dash(self, path_prefix):
        app = Dash(__name__,
                   requests_pathname_prefix=path_prefix,
                   title='FeatureCloud Cluster Visualization App',
                   suppress_callback_exceptions=True)
        if len(self.dataframes_by_k_value) == 0 and len(self.volcano_df) == 0:
            f = open('README.md', 'r')
            app.layout = html.Div([
                dbc.Row(
                    dbc.Col(width=8, children=[
                        dbc.Toast(
                            [html.P('Local data cannot be found.', className="mb-0")],
                            id="data-validation-toast",
                            header="Error",
                            duration=10000,
                            is_open=True,
                            icon="danger",
                            style={"position": "fixed", "top": 66, "right": 10, "width": 350},
                        ),
                        dcc.Markdown(f.read())
                    ]),
                    justify='center'
                )
            ],
                className='help-ct')
        else:
            confounding_style = {'display': 'none'} if len(self.dataframes_by_k_value) == 0 else {}
            distance_style = {'display': 'none'} if len(self.distance_df) == 0 else {}
            scree_plot_style = {'display': 'none'} if len(self.scree_plot) == 0 else {}
            cluster_quality_style = {'display': 'none'} if len(self.k_values) <= 1 else {}
            volcano_plot_style = {'display': 'none'} if len(self.volcano_df) <= 1 else {}
            finished_button_style = {'display': 'none'} if self.env != 'fc' else {'float': 'right'}
            tab_value = 'tab-confounders' if len(self.dataframes_by_k_value) > 0 else 'tab-volcano-plot'

            app.layout = html.Div([
                html.H2('FeatureCloud Cluster Visualization App', className='fc-header'),
                dbc.Button('Finish', id='btn-finished', color='primary', className='me-1',
                           style=finished_button_style,
                           title='Finished visualization, stop app and proceed to next step if any.'),
                dbc.Toast(
                    [html.P('Visualization finished. Will proceed to next step or finish the workflow.',
                            className="mb-0")],
                    id="toaster-visualization-finished",
                    header="Visualization",
                    duration=5000,
                    is_open=False,
                    style={"position": "fixed", "top": 66, "right": 10, "width": 350},
                ),
                dcc.Tabs(id="tabs-ct", value=tab_value, children=[
                    dcc.Tab(label='Confounders', value='tab-confounders', style=confounding_style),
                    dcc.Tab(label='Distances', value='tab-distances', style=distance_style),
                    dcc.Tab(label='Clustering Quality', value='tab-clustering-quality', style=cluster_quality_style),
                    dcc.Tab(label='Scree plot', value='tab-scree-plot', style=scree_plot_style),
                    dcc.Tab(label='Volcano plot', value='tab-volcano-plot', style=volcano_plot_style),
                    dcc.Tab(label='Help', value='tab-help'),
                ]),
                html.Div(id='tabs-content-ct', style={'width': '75%', 'margin': '0 auto'}),
                dbc.Toast(
                    [html.P(self.data_errors, className="mb-0")],
                    id="data-validation-toast",
                    header="Error",
                    duration=10000,
                    is_open=False,
                    icon="danger",
                    style={"position": "fixed", "top": 66, "right": 10, "width": 350},
                ),
            ])

        @app.callback(
            Output('tabs-content-ct', 'children'),
            Output('data-validation-toast', 'is_open'),
            Input('tabs-ct', 'value')
        )
        def render_content(tab):
            show_toast = False
            if len(self.data_errors) > 0:
                show_toast = True
                self.data_errors = ''
            if tab == 'tab-confounders':
                return render_confounders(), show_toast
            elif tab == 'tab-distances':
                return self.render_distances(), show_toast
            elif tab == 'tab-clustering-quality':
                return self.render_clustering_quality(), show_toast
            elif tab == 'tab-scree-plot':
                return self.render_scree_plot(), show_toast
            elif tab == 'tab-volcano-plot':
                return self.render_volcano_plot(), show_toast
            elif tab == 'tab-help':
                return self.render_help(), show_toast

        @app.callback(
            Output('toaster-visualization-finished', 'is_open'),
            Input('btn-finished', 'n_clicks'),
        )
        def set_finished(n_clicks):
            if n_clicks is not None and n_clicks > 0:
                if self.callback_fn_terminal_state is not None:
                    self.callback_fn_terminal_state()
                else:
                    # Stopping Dash
                    func = request.environ.get('werkzeug.server.shutdown')
                    if func is None:
                        raise RuntimeError('Not running with the Werkzeug Server')
                    func()
                return True
            return False

        def render_confounders():
            confounding_df = self.get_df_by_k_value(self.k_values[0], self.dataframes_by_k_value)
            datatable_columns = confounding_df.columns.to_list()
            height_multiplier = 0 if len(self.confounding_meta) == 0 else len(self.confounding_meta.index)
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
                                dbc.Col(children=self.get_k_filter(id_post_tag)),
                                dbc.Col(self.get_cluster_values_filter(id_post_tag)),
                                dbc.Col(
                                    children=
                                    [
                                        html.Span('X axes', style={'float': 'left', 'margin-top': '5px'}),
                                        html.Span(
                                            dcc.Dropdown(self.data_columns, self.data_columns[0], id='xaxis-dropdown',
                                                         className='fc-dropdown', clearable=False,
                                                         style={'float': 'left'})),
                                        html.Span('Y axes',
                                                  style={'float': 'left', 'margin-top': '5px', 'margin-left': '10px'}),
                                        html.Span(
                                            dcc.Dropdown(self.data_columns, self.data_columns[1], id='yaxis-dropdown',
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
                            self.get_confounding_factors_filter('confounders'),
                            id='confounding-factors-filter-ct',
                            className='confounding-factors-filter-ct',
                            style={'height': confounders_filter_height}
                        ),
                    )
                ),
                self.get_download_button(),
                dbc.Row(dcc.Graph(id='confounders-scatter', className='confounders-scatter')),
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
            if use_clusters:
                clustering_field = 'cluster'
            else:
                clustering_field = 'client_id'
            confounding_df = self.get_df_by_k_value(k_value, self.dataframes_by_k_value)
            cluster_checklist_values = self.get_cluster_values_list(confounding_df)
            # Detect if K value has changed, to reset checklist values to all values selected
            if self.k_value_confounders != k_value:
                selected_clusters = cluster_checklist_values
            self.k_value_confounders = k_value
            # filter base dataframe
            index_list = self.filter_dataframe_on_confounding_factors(confounding_df, selected_clusters,
                                                                      checklist_values,
                                                                      range_values, use_clusters)
            confounding_df = confounding_df[confounding_df.index.isin(index_list)]
            fig = get_figure_with_subplots(confounding_df, k_value, xaxis, yaxis, use_pie_charts, clustering_field)

            self.save_fig_as_image(fig)

            return fig, cluster_checklist_values, selected_clusters

        def get_figure_with_subplots(confounding_df, k_value, xaxis, yaxis, use_pie_charts, clustering_field):
            cluster_values_list = confounding_df[clustering_field].unique()
            cluster_values_list_length = len(cluster_values_list)
            nr_of_confounding_factors = 0 if len(self.confounding_meta) == 0 else len(self.confounding_meta.index)

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
                    nr_rows = nr_of_confounding_factors + cluster_values_list_length + 1
                else:
                    nr_rows = nr_of_confounding_factors + k_value + 1

            specs, subplot_titles = self.get_specs_for_matrix(nr_rows, nr_cols, use_pie_charts, clustering_field)
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

                if cluster_values_list_length > 1:
                    path = self.confidence_ellipse(df[xaxis], df[yaxis])
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
            for i in range(1, cluster_values_list_length + 1):
                color = DEFAULT_PLOTLY_COLORS[i]
                df = confounding_df[confounding_df[clustering_field] == i]
                if (
                        cluster_values_list_length > 1 and clustering_field == 'cluster') or clustering_field == 'client_id':
                    for j in range(0, nr_of_confounding_factors):
                        col = self.confounding_meta.iloc[j]['name']
                        data_type = self.confounding_meta.iloc[j]['data_type']
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
                col = self.confounding_meta.iloc[j]['name']
                data_type = self.confounding_meta.iloc[j]['data_type']
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
            fig.update_layout(modebar_remove=['toImage'])
            return fig

        @app.callback(
            Output("download-plot", "data"),
            [Input("btn-download-plot", "n_clicks")]
        )
        def download_image(n_clicks):
            if n_clicks is not None and n_clicks > 0:
                return dcc.send_file(os.path.join(self.download_dir, 'plot.png'))

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
            confounding_df = self.get_df_by_k_value(k_value, self.dataframes_by_k_value)
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
                df = pd.DataFrame(data=self.filter_dataframe_inverse_on_id(k_value, df['id'].tolist()), columns=columns)

            if group_name is None:
                group_name = default_file_name
            else:
                group_name = group_name.strip()
                if len(group_name) == 0:
                    group_name = default_file_name

            # Save data in file system as well
            df.to_csv(os.path.join(self.download_dir, f'{group_name}.csv'))

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
            confounding_df = self.get_df_by_k_value(k_value, self.dataframes_by_k_value)
            cluster_checklist_values = self.get_cluster_values_list(confounding_df)
            # Detect if K value has changed, to reset checklist values to all values selected
            if self.k_value_distance != k_value:
                selected_clusters = cluster_checklist_values
            self.k_value_distance = k_value
            index_list = self.filter_dataframe_on_confounding_factors(confounding_df, selected_clusters,
                                                                      checklist_values, range_values, True)
            display_error_toaster = False
            if len(index_list) == 0:
                display_error_toaster = True
                index_list = self.heatmap_index_list
            else:
                self.heatmap_index_list = index_list

            df = self.distance_df[self.distance_df.index.isin(index_list)]
            fig = dash_bio.Clustergram(
                data=df,
                column_labels=list(df.columns.values),
                row_labels=list(df.index),
                height=800,
                width=1400,
                hidden_labels='rows, columns'
            )
            fig.update_layout(modebar_remove=['toImage'])

            self.save_fig_as_image(fig)

            return fig, cluster_checklist_values, selected_clusters, display_error_toaster

        @app.callback(
            Output('cluster_quality_graph', 'figure'),
            Input('k-labels', 'value')
        )
        def filter_k_label(value):
            df = self.get_df_by_k_value(value, self.silhouette)
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
            fig.update_layout(modebar_remove=['toImage'])

            self.save_fig_as_image(fig)

            return fig

        @app.callback(
            Output('dashbio-default-volcanoplot', 'figure'),
            Input('effect-size-input', 'value'),
            Input('genome-wide-line-input-button', 'n_clicks'),
            State('genome-wide-line-input', 'value')
        )
        def update_volcanoplot(effects, n_clicks, genomewideline):
            fig = dash_bio.VolcanoPlot(
                dataframe=self.volcano_df,
                genomewideline_value=genomewideline,
                effect_size_line=effects
            )
            # .update_traces(mode='markers+text', selector=dict(marker_color='red'))\
            # .update_traces(text=hover_text, selector=dict(marker_color='red')) \
            # .update_traces(textposition='top center', selector=dict(marker_color='red'))
            self.save_fig_as_image(fig)
            return fig

        return app

    def save_fig_as_image(self, fig):
        # Save image to download folder to be available for download
        filepath = os.path.join(self.download_dir, 'plot.png')
        with open(filepath, "wb") as fp:
            fp.write(fig.to_image(width=1920, height=1080))

    def filter_dataframe_on_confounding_factors(self, confounding_df, selected_clusters, checklist_values, range_values,
                                                use_clusters):
        selected_cluster_ids = []
        if len(self.confounding_meta) == 0:
            return confounding_df.index.tolist()
        if len(selected_clusters) > 0:
            for cluster_value in selected_clusters:
                cluster_id = int(cluster_value.split()[1])
                selected_cluster_ids.append(cluster_id)
        if use_clusters:
            confounding_df = confounding_df.loc[confounding_df['cluster'].isin(selected_cluster_ids)]

        confounding_length = len(self.confounding_meta.index)
        # Filter data based on active filters
        checklist_index = range_index = 0
        for j in range(0, confounding_length):
            col = self.confounding_meta.iloc[j]["name"]
            data_type = self.confounding_meta.iloc[j]['data_type']
            if data_type == 'continuous':
                range_list = range_values[range_index]
                confounding_df = confounding_df.loc[confounding_df[col].between(range_list[0], range_list[1])]
                range_index += 1
            elif data_type == 'discrete' or data_type == 'ordinal':
                checklist = checklist_values[checklist_index]
                confounding_df = confounding_df.loc[confounding_df[col].isin(checklist)]
                checklist_index += 1
        return confounding_df.index.tolist()

    def filter_dataframe_inverse_on_id(self, k_value, selected_ids):
        confounding_df = self.get_df_by_k_value(k_value, self.dataframes_by_k_value)
        selected_data = confounding_df.loc[~confounding_df['id'].isin(selected_ids)]
        return selected_data

    def render_clustering_quality(self):
        return html.Div([
            dbc.Row([
                dbc.Col(children=
                [
                    html.Span('K', style={'float': 'left', 'margin-top': '6px'}),
                    html.Span(
                        dcc.Dropdown(self.k_values, self.k_values[0], id='k-labels', className='fc-dropdown',
                                     clearable=False,
                                     style={'float': 'left'}))
                ],
                    style={'height': '60px', 'width': '100px', 'margin': '20px 70px'}
                ),
            ]),
            self.get_download_button(),
            dbc.Row([
                dcc.Graph(id="cluster_quality_graph"),
            ])
        ])

    def render_scree_plot(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            mode='lines+markers',
            x=self.scree_plot['component'],
            y=self.scree_plot['eigenvalue'],
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

        fig.update_layout(modebar_remove=['toImage'])
        self.save_fig_as_image(fig)

        return html.Div([
            self.get_download_button(),
            dbc.Row(
                dcc.Graph(
                    id='scree-plot',
                    figure=fig
                )
            )],
            style={'margin-top': '25px'})

    def render_volcano_plot(self):
        min_effect = math.floor(self.volcano_df['EFFECTSIZE'].min())
        max_effect = math.ceil(self.volcano_df['EFFECTSIZE'].max())
        min_effect_value = math.floor(min_effect + 0.3 * (max_effect - min_effect))
        max_effect_value = math.ceil(max_effect - 0.3 * (max_effect - min_effect))

        min_p_value = -math.floor(np.log10(self.volcano_df['P'].min()))
        max_p_value = -math.ceil(np.log10(self.volcano_df['P'].max()))
        min_genome_wide_line = min(min_p_value, max_p_value)
        max_genome_wide_line = max(min_p_value, max_p_value)
        genome_wide_line_value = math.floor(min_genome_wide_line + 0.3 * (max_genome_wide_line - min_genome_wide_line))

        fig = dash_bio.VolcanoPlot(
            dataframe=self.volcano_df,
            genomewideline_value=genome_wide_line_value,
        )
        self.save_fig_as_image(fig)

        return html.Div([
            self.get_download_button(),
            dbc.Row(children=[
                dbc.Label('Effect sizes'),
                dcc.RangeSlider(
                    id='effect-size-input',
                    min=min_effect,
                    max=max_effect,
                    step=0.05,
                    marks={i: {'label': str(i)} for i in range(min_effect, max_effect)},
                    value=[min_effect_value, max_effect_value]
                )],
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Label("Threshold"),
                ),
            ),
            dbc.Row(children=[
                html.Span(
                    dbc.Input(
                        id='genome-wide-line-input',
                        type='number',
                        min=min_genome_wide_line,
                        max=max_genome_wide_line,
                        value=genome_wide_line_value,
                        style={"width": 75},
                    ),
                    style={"width": 80, 'float': 'left'},
                ),
                html.Span(
                    dbc.Button(
                        "Set",
                        id='genome-wide-line-input-button',
                        n_clicks=0,
                    ),
                    style={"width": 80, 'float': 'left'},
                ),
            ]),
            dbc.Row(
                dcc.Graph(
                    id='dashbio-default-volcanoplot',
                    figure=fig
                )
            )
        ],
            style={'margin-top': '25px'}
        )

    def render_help(self):
        f = open('README.md', 'r')
        return dcc.Markdown(f.read(), className='help-ct')

    def get_download_button(self):
        return dbc.Row(
            html.Div(
                [dbc.Button("Download plot as image", id="btn-download-plot", size='sm'), Download(id="download-plot")],
                style={'width': '200px', 'float': 'right'}
            )
        )

    def render_distances(self):
        id_post_tag = 'distances-tab'
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(children=self.get_k_filter(id_post_tag)),
                        dbc.Col(self.get_cluster_values_filter(id_post_tag)),
                    ]
                ),
                dbc.Row(
                    dbc.Col(
                        html.Div(
                            children=self.get_confounding_factors_filter('distance'),
                            style={'margin-top': '20px'}
                        )
                    )
                ),
                self.get_download_button(),
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

    def confidence_ellipse(self, x, y, n_std=1.96, size=100):
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

    def get_specs_for_matrix(self, rows, cols, use_pie_charts, clustering_field):
        specs = []
        subplot_titles = []
        nr_of_confounding_factors = 0 if len(self.confounding_meta) == 0 else len(self.confounding_meta.index)
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
                            {'type': 'pie' if self.confounding_meta.iloc[j][
                                                  'data_type'] == 'discrete' and use_pie_charts else 'xy'})
                        if (len(
                                self.k_values) > 1 and clustering_field == 'cluster') or clustering_field == 'client_id':
                            title = f'{clustering_field.capitalize()} {i - cols}: {self.confounding_meta.iloc[j]["name"].capitalize()}'
                        else:
                            title = ''
                    else:
                        current_specs_row.append({'type': 'xy'})
                        if (len(
                                self.k_values) > 1 and clustering_field == 'cluster') or clustering_field == 'client_id':
                            title = f'All {clustering_field}s: {self.confounding_meta.iloc[j]["name"].capitalize()}'
                        else:
                            title = self.confounding_meta.iloc[j]["name"].capitalize()
                    subplot_titles.append(title)
            specs.append(current_specs_row)
        return specs, subplot_titles

    def get_confounding_factors_filter(self, id_pre_tag):
        html_elem_list = []
        if len(self.confounding_meta) == 0:
            return html_elem_list
        confounding_df = self.get_df_by_k_value(self.k_values[0], self.dataframes_by_k_value)
        confounding_length = len(self.confounding_meta.index)
        confounding_base_length = len(self.confounding_meta)
        confounding_selector_options = []
        for j in range(0, confounding_base_length):
            confounding_selector_options.append(self.confounding_meta.iloc[j]['name'].capitalize())

        html_elem_list.append(
            dbc.Row(
                children=
                [
                    dbc.Col(html.H5("Confounding factors filter")),
                ]
            )
        )
        for j in range(0, confounding_length):
            col = self.confounding_meta.iloc[j]["name"]
            data_type = self.confounding_meta.iloc[j]['data_type']
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

    def get_cluster_values_list(self, df):
        cluster_values = df.cluster.unique()
        cluster_values_list = []
        for i in cluster_values:
            cluster_values_list.append(f'Cluster {i}')

        return cluster_values_list

    def get_client_values_list(self, df):
        client_values = df.client_id.unique()
        client_values_list = []
        for i in client_values:
            client_values_list.append(f'Client {i}')

        return client_values_list

    def get_df_by_k_value(self, k_value, base_obj):
        for k_obj in base_obj:
            if k_obj['k'] == k_value:
                return k_obj['df']
        return []

    def get_k_filter(self, id_post_tag):
        disable_select = False
        if len(self.k_values) == 1:
            disable_select = True
        return [
            html.Span('K', style={'float': 'left', 'margin-top': '5px'}),
            html.Span(
                dcc.Dropdown(self.k_values, self.k_values[0], id=f'k-filter-{id_post_tag}', className='fc-dropdown',
                             disabled=disable_select, clearable=False, style={'float': 'left', 'margin-right': '15%'})),
        ]

    def get_cluster_values_filter(self, id_post_tag):
        confounding_df = self.get_df_by_k_value(self.k_values[0], self.dataframes_by_k_value)
        cluster_values_list = self.get_cluster_values_list(confounding_df)
        style = {}
        if len(self.k_values) == 1:
            style = {'display': 'none'}
        return html.Span(
            dcc.Checklist(cluster_values_list, cluster_values_list, style=style,
                          inline=True, id=f'cluster-values-checklist-{id_post_tag}', className="fc-checklist"),
        )

    def get_client_values_filter(self, id_post_tag):
        confounding_df = self.get_df_by_k_value(self.k_values[0], self.dataframes_by_k_value)
        cluster_values_list = self.get_client_values_list(confounding_df)
        return html.Span(
            dcc.Checklist(cluster_values_list, cluster_values_list,
                          inline=True, id=f'client-values-checklist-{id_post_tag}', className="fc-checklist"),
        )
