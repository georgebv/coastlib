import warnings
import pint
import pandas as pd
from coastlib.design.design_data.section_aisc import AiscSection


ureg = pint.UnitRegistry()
sec = AiscSection('hss14x4x3/8')


class Column:
    
    def __init__(self, section, **kwargs):
        
        self.section = section
    
    def __repr__(self):
        
        return self.report()

    def test(self, parameters):

        pass

    def report(self):
        
        # Report on the beam - echo its type, section parameters, type-specific parameters, applied forces and conditions,
        # and results on passed tests

        pass


class Cantilever:
    
    def __init__(self, section, **kwargs):
        
        self.section = section
        

class Beam:
    
    def __init__(self, section, **kwargs):
        
        self.section = section
    
