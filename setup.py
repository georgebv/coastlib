from setuptools import setup, find_packages
import os


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='coastlib',
    version='1.0.0',
    author='Georgii Bocharov',
    author_email='bocharovgeorgii@gmail.com',
    description='Coastal engineering library and tools',
    long_description=long_description,
    url='https://github.com/georgebv/coastlib',
    license='GPLv3',
    keywords='coastal ocean marine engineering',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ]
)
