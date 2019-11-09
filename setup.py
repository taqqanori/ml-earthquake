from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ml-earthquake',
    version='0.0.1',
    description='predict earthpuakes by machine learning',
    long_description=readme,
    author='taqqanori',
    author_email='taqanori@gmail.com',
    url='https://github.com/taqqanori/ml-earthquake',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
