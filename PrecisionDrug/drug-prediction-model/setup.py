from setuptools import setup, find_packages

setup(
    name='drug-prediction-model',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project to build a drug prediction model using Decision Tree Classifier.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'flask',  # if using Flask for the API
        'jupyter',  # if using Jupyter notebooks
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)