import os
import time
from threading import Thread

from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.engine.app import app
from bottle import Bottle

import fcvisualization
import plotly.express as px
import states

server = Bottle()

"""
Normally the app is started within FeatureCloud environment ('fc'),
but during development it's useful to be able to start it outside FeatureCloud, using included data ('native')
We can detect the environment from the presence of the environment variable PATH_PREFIX. 
It can be hard-coded also, if needed.
"""
path_prefix = os.getenv("PATH_PREFIX")
env = 'fc' if path_prefix else 'native'  # 'native', 'fc'
extra_visualization_content = []

def start_app():
    app.register()
    server.mount('/api', api_server)
    server.run(host='localhost', port=5000)


if __name__ == '__main__':
    if env == 'fc':
        print("Starting visualization app in fc mode")
        start_app()
    else:
        print("Starting visualization app in native mode")
        df = px.data.iris()  # iris is a pandas DataFrame
        fig = px.scatter(df, x="sepal_width", y="sepal_length")
        fig2 = px.scatter(df, x="sepal_length", y="sepal_width")
        extra_visualization_content.append({
            "title": "My Diagram",
            "fig": fig,
        })
        fc_visualization = fcvisualization.fcvisualization()
        fc_visualization.start(env, None, None, extra_visualization_content)
