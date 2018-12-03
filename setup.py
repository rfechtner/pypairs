from setuptools import setup
from io import open
import versioneer

with open('requirements.txt', encoding='utf-8') as requirements:
    requires = [l.strip() for l in requirements]

with open('README.rst', encoding='utf-8') as readme_f:
    readme = readme_f.read()

setup(
    name='pypairs',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A python scRNA-Seq classifier',
    long_description=readme,
    packages=['pypairs'],
    url='https://github.com/rfechtner/pypairs',
    license='BSD',
    author='Ron Fechtner',
    author_email='ronfechtner@gmail.com',
    python_requires='>=3.5',
    install_requires=requires,
    extras_require=dict(
        plotting=['matplotlib', 'plotly'],
        scanpy=['scanpy']
    )
)
