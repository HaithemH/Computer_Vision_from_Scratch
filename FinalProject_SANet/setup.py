from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.0.1'

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='sanet',
    version=__version__,
    description='Tensorflow implementation of SANet.',
    url='https://github.com/lev1khachatryan/ASDS_CV/tree/master/FinalProject_SANet',
    download_url='https://github.com/lev1khachatryan/ASDS_CV.git',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    author='Levon Khachatryan',
    install_requires=install_requires,
    setup_requires=['numpy>=1.10', 'scipy>=0.17'],
    dependency_links=dependency_links,
    author_email='levon.khachatryan.1996.db@gmail.com'
)