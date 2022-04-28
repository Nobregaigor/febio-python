import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="febio-python",
    version="0.1.4",
    author="Igor N",
    description="Collection of python modules to work with FEBio.",
    url="https://github.com/Nobregaigor/febio-python",
    packages=setuptools.find_namespace_packages(include=["febio_python.*"]),
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['febio-python=febio_python.cli.main:main'],
    },
    # include_package_data=True
    package_data={'': ['*.json']}
)
