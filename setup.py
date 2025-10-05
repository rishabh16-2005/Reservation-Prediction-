from setuptools import setup,find_packages
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name = 'Hotel Reserrvation',
    version = '0.1',
    author='Rishabh Galave',
    packages=find_packages(),
    install_requires = requirements
)