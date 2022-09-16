import os

from FeatureCloud.app.engine.app import AppState, app_state

import fcvisualization

TERMINAL = False

def callbackFnTerminalState():
    global TERMINAL
    print("Transition to terminal state triggered...")
    TERMINAL = True


@app_state('initial')
class InitialState(AppState):
    def register(self):
        print('registering transition from initial to plot')
        self.register_transition('plot')

    def run(self) -> str:
        path_prefix = os.getenv("PATH_PREFIX")
        print("PATH_PREFIX environment variable: ", path_prefix)
        print('Plot start...')
        fc_visualization = fcvisualization()
        fc_visualization.start('fc', path_prefix, callbackFnTerminalState)
        return 'plot'


@app_state('plot')
class PlotState(AppState):
    def register(self):
        print('register transitions from plot state to terminal')
        self.register_transition('terminal')

    def run(self) -> str:
        # @TODO implement this using threads that can intercommunicate with each other: https://github.com/plotly/dash-core-components/issues/952
        # This will be needed when other states intervene
        if TERMINAL is True:
            print('plot is finished')
            return 'terminal'
        return 'plot'
