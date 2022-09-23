import multiprocessing
import os
import time

from FeatureCloud.app.engine.app import AppState, app_state

import fcvisualization
import plotly.express as px

TERMINAL = False
fc_visualization = fcvisualization.fcvisualization()
extra_visualization_content = []


def callback_fn_terminal_state():
    global TERMINAL
    print("Transition to terminal state triggered...")
    TERMINAL = True


@app_state('initial')
class InitialState(AppState):
    def register(self):
        print('registering transition from initial to plot')
        self.register_transition('plot')

    def run(self) -> str:
        global extra_visualization_content
        path_prefix = os.getenv("PATH_PREFIX")
        print("PATH_PREFIX environment variable: ", path_prefix)
        print('Plot start...')
        process = multiprocessing.Process(target=fc_visualization.start, args=('fc', path_prefix, callback_fn_terminal_state, []))
        process.start()
        time.sleep(15)
        print("Stopping")
        process.terminate()
        print("Starting new diagram")
        df = px.data.iris()  # iris is a pandas DataFrame
        fig = px.scatter(df, x="sepal_width", y="sepal_length")
        fig2 = px.scatter(df, x="sepal_length", y="sepal_width")
        extra_visualization_content.append({
            "title": "My Diagram",
            "fig": fig,
        })
        extra_visualization_content.append({
            "title": "My Diagram 2",
            "fig": fig2,
        })
        process = multiprocessing.Process(target=fc_visualization.start,
                                          args=('fc', path_prefix, callback_fn_terminal_state, extra_visualization_content))
        process.start()

        return 'plot'


@app_state('plot')
class PlotState(AppState):
    def register(self):
        print('register transitions from plot state to terminal')
        self.register_transition('plot')
        self.register_transition('terminal')

    def run(self) -> str:
        # This will be needed when other states intervene
        if TERMINAL is True:
            print('plot is finished')
            return 'terminal'
        print('plot is running')
        return 'plot'
