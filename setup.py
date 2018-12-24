from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='coastlib',
    version='0.8',
    description='Coastal engineering library and tools',
    long_description=long_description,
    author='Georgii Bocharov',
    license='GPLv3',
    author_email='bocharovgeorgii@gmail.com',
    url='https://github.com/georgebv/coastlib',
    keywords=['coastal', 'ocean', 'marine', 'engineering'],
    packages=find_packages(exclude=['tests', 'docs']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.6',
    ],
    zip_safe=False,
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas']
)
