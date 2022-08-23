import os

from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.engine.app import app
from bottle import Bottle

import states
import plot
import plotly.io as pio

pio.kaleido.scope.chromium_args = tuple([arg for arg in pio.kaleido.scope.chromium_args if arg != "--disable-dev-shm-usage"])


"""
Normally the app is started within FeatureCloud environment ('fc'),
but during development it's useful to be able to start it outside FeatureCloud, using included data ('native')
We can detect the environment from the presence of the environment variable PATH_PREFIX. 
It can be hard-coded also, if needed.
"""
path_prefix = os.getenv("PATH_PREFIX")
env = 'fc' if path_prefix else 'native'  # 'native', 'fc'


def start_app():
    app.register()
    server = Bottle()
    server.mount('/api', api_server)
    server.run(host='localhost', port=5000)


if __name__ == '__main__':
    if env == 'fc':
        start_app()
    else:
        plot.start(env, None)
