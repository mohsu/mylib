import setuptools
from setuptools import setup

setup(
    name='mylib',
    version='1.0',
    packages=setuptools.find_packages(),
    url='https://github.com/mohsu/mylib',
    license='',
    author='Maureen Hsu',
    author_email='',
    description='',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['python=3.7.7', 'tensorflow=2.1.0', 'pynvml', 'requests', 'loguru=0.4.1', 'imgaug=0.4.0',
                      'numpy=1.18.1', 'matplotlib=3.2.1', 'opencv=4.2.0'],
    python_requires='>=3.7',
)