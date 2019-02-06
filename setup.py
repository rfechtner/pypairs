import sys
if sys.version_info < (3,):
    sys.exit('scanpy requires Python >= 3.5')

from setuptools import setup, find_packages
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
    packages=find_packages(),
    package_data = {
        '': ['*.json', '*.gz'],
    },
    url='https://github.com/rfechtner/pypairs',
    license='BSD',
    author='Ron Fechtner',
    author_email='ronfechtner@gmail.com',
    python_requires='>=3.5',
    install_requires=requires,
    extras_require=dict(
        plotting=['matplotlib', 'plotly'],
        scanpy=['scanpy']
    ),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Framework :: Jupyter',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Visualization',
    ]
)
