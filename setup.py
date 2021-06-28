from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'The GBRSs Module Series'
LONG_DESCRIPTION = 'This package is supposed to allow the running of all GBRS models.'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="GBRSs", 
        version=VERSION,
        author="Longyin Cui",
        author_email="<cuilongyin@hotmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['surprise','pandas','numpy','collections'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'recommender systems', 'privacy preserving'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)