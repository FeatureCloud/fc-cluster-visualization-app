import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name="FeatureCloudVisualization",
                 version="0.0.2",
                 license='MIT',
                 author="FeatureCloud",
                 author_email="balazs.orban@gnd.ro",
                 description="FeatureCloud Visualization",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://github.com/FeatureCloud/fc-cluster-visualization-app",
                 project_urls={
                     "Bug Tracker": "https://github.com/FeatureCloud/fc-cluster-visualization-app/issues",
                 },
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "Operating System :: OS Independent",
                 ],
                 packages=setuptools.find_packages(include=['featurecloudvisualization', 'featurecloudvisualization.*']),
                 python_requires=">=3.7",
                 include_package_data=True,
                 package_data={'': ['FeatureCloudVisualization/*']},
                 install_requires=['attrs==21.4.0', 'biopython==1.79', 'Brotli==1.0.9', 'certifi==2021.10.8',
                                   'charset-normalizer==2.0.12', 'click==8.0.4', 'colour==0.1.5', 'cycler==0.11.0',
                                   'dash==2.2.0', 'dash-bio==1.0.1', 'dash-bootstrap-components==1.0.3',
                                   'dash-core-components==2.0.0', 'dash-daq==0.5.0', 'dash-html-components==2.0.0',
                                   'dash-table==5.0.0', 'Flask==2.0.3', 'Flask-Compress==1.11', 'fonttools==4.30.0',
                                   'GEOparse==2.0.3', 'idna==3.3', 'itsdangerous==2.1.0', 'Jinja2==3.0.3',
                                   'joblib==1.1.0', 'jsonschema==4.4.0', 'kaleido==0.2.1', 'kiwisolver==1.3.2',
                                   'MarkupSafe==2.1.0', 'matplotlib==3.5.1', 'numpy==1.22.2', 'packaging==21.3',
                                   'pandas==1.4.1', 'ParmEd==3.4.3', 'periodictable==1.6.0', 'Pillow==9.0.1',
                                   'plotly==5.6.0', 'pyparsing==3.0.7', 'pyrsistent==0.18.1', 'python-dateutil==2.8.2',
                                   'pytz==2021.3', 'requests==2.27.1', 'scikit-learn==1.0.2', 'scipy==1.8.0',
                                   'six==1.16.0', 'tenacity==8.0.1', 'threadpoolctl==3.1.0', 'tqdm==4.64.0',
                                   'urllib3==1.26.9', 'Werkzeug==2.0.3', 'yellowbrick==1.4']

                 )
