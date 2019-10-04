from setuptools import setup
from os import path

DIR = path.dirname(path.abspath(__file__))
#INSTALL_PACKAGES = open(path.join(DIR, 'requirements.txt')).read().splitlines()

with open(path.join(DIR, 'README.md')) as f:
    README = f.read()

setup(
    name='reclib',
    packages=['reclib'],
    description="A Python Library for Recommender System",
    long_description=README,
    long_description_content_type='text/markdown',
    #install_requires=INSTALL_PACKAGES,
    install_requires=[
            'tqdm',
            'lmdb',
            'numpy',
            'numpydoc>=0.8.0',
            'torch>=1.1.0',
            'scikit-learn',
            'pandas',
            'assertpy',
            'overrides'
    ],
    version='0.2.8',
    url='https://github.com/tingkai-zhang/reclib',
    author='Tingkai Zhang',
    author_email='tingkai.zhang@outlook.com',
    keywords=['recommender-system', 'machine-learning', 'deep-learning'],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pytest-sugar'
    ],
    package_data={
        # include json and pkl files
        '': ['*.json', 'models/*.pkl', 'models/*.json'],
    },
    include_package_data=True,
    python_requires='>=3.6'
)