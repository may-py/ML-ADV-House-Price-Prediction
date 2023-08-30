from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT="-e ."
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        
        #Replace /n to blank that comes from lines while reading
        requirements=[req.replace("\n","") for req in requirements]  

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="Advance House Price Prediction Project",
    version="1.0.0",
    author="Mayur",
    author_email="mayur1601@yahoo.in",
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt') #['pandas','numpy','seaborn'],
)