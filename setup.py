from setuptools import setup, find_packages

setup(
    name='mylib_maureen',
    version='2.0.0',
    packages=find_packages(),
    url='https://github.com/mohsu/mylib',
    author='Maureen Hsu',
    description='my utils for cross-project use',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pynvml', 'requests', 'loguru', 'numpy==1.18.1', 'matplotlib', 'configobj'],
    python_requires='>=3.7',
)