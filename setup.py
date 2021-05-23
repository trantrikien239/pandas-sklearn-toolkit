from setuptools import setup, find_packages

__PKG_NAME__ = 'pskit'
__PKG_VERSION__ = '0.0.5'


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name=__PKG_NAME__,
    version=__PKG_VERSION__,
    description="Toolkit for an easy use of pandas, sklearn and popular ml libraries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trantrikien239/pandas-sklearn-toolkit",
    author="Kien Tran",
    author_email="trantrikien239@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    install_requires=required
)
