import os

from FeatureCloud.app.engine.app import AppState, app_state

import plot


@app_state('initial')
class InitialState(AppState):
    def register(self):
        self.register_transition('plot')

    def run(self) -> str:
        path_prefix = os.getenv("PATH_PREFIX")
        print("PATH_PREFIX environment variable: ", path_prefix)
        plot.start('fc', path_prefix)
        return 'plot'


@app_state('plot')
class PlotState(AppState):
    def register(self):
        self.register_transition('plot')
        self.register_transition('terminal')

    def run(self) -> str:
        return 'plot'
