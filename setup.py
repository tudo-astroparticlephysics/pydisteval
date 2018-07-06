# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='disteval',
    version='0.0.1',

    description='A package to investigate missmatches and agreement in '
                'multivariate parameter distributions',
    long_description=long_description,

    url='https://github.com/tudo-astroparticlephysics/pydisteval',

    author='Mathis Boerner, Jens Buss',
    author_email='mathis.boerner@tu-dortmund.de, jens.buss@tu-dortmund.de',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
    ],
    # What does your project relate to?
    keywords='multivariate distribution evaluation',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib>=1.4',
        'scikit-learn>=0.18.1',
        'tables',
        'h5py',
        'scipy',
        'futures'],

    entry_points={
        'console_scripts': ['fact_example=examples.fact_example.py:main'],
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=3.1.0', 'setuptools>=34.4.1'],
)
