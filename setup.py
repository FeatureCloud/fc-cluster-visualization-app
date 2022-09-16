import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name="fcvisualization",
                 version="0.0.0.2",
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
                 packages=setuptools.find_packages(include=['fcvisualization', 'fcvisualization.*']),
                 python_requires=">=3.7",
                 include_package_data=True,
                 package_data={'': ['./fcvisualization/*']},
                 install_requires=['attrs', 'biopython', 'Brotli', 'certifi',
                                   'charset-normalizer', 'click', 'colour', 'cycler',
                                   'dash', 'dash-bio', 'dash-bootstrap-components',
                                   'dash-core-components', 'dash-daq', 'dash-html-components',
                                   'dash-table', 'Flask', 'Flask-Compress', 'fonttools',
                                   'GEOparse', 'idna', 'itsdangerous', 'Jinja2',
                                   'joblib', 'jsonschema', 'kaleido', 'kiwisolver',
                                   'MarkupSafe', 'matplotlib', 'numpy', 'packaging',
                                   'pandas', 'ParmEd', 'periodictable', 'Pillow',
                                   'plotly', 'pyparsing', 'pyrsistent', 'python-dateutil',
                                   'pytz', 'requests', 'scikit-learn', 'scipy',
                                   'six', 'tenacity', 'threadpoolctl', 'tqdm',
                                   'urllib3', 'Werkzeug', 'yellowbrick']

                 )
