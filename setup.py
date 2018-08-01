from setuptools import setup, find_packages

setup(
    name='coastlib',
    version='0.2',
    description='Coastal engineering library and tools',
    author='Georgii Bocharov',
    license='GPLv3',
    author_email='bocharovgeorgii@gmail.com',
    url='https://github.com/georgebv/coastlib',
    keywords=['coastal', 'ocean', 'marine'],
    packages=find_packages(exclude=['tests', 'docs']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Engineers',
        'Topic :: Coastal engineering',
        'License :: OSI Approved :: GPLv3 License',
        'Programming Language :: Python :: 3.6.2',
    ],
    zip_safe=False, install_requires=['numpy', 'scipy', 'matplotlib', 'pandas']
)
