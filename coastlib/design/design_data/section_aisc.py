import pandas as pd
import os
import pint


ureg = pint.UnitRegistry()
dir_path = os.path.dirname(os.path.realpath(__file__))
sections_us = pd.read_pickle(os.path.join(dir_path, 'aisc_shapes_db_14.1_us.pyc'))
sections_si = pd.read_pickle(os.path.join(dir_path, 'aisc_shapes_db_14.1_si.pyc'))


# data_1 = pd.read_excel(os.path.join(r'C:\Users\GRBH\Desktop', r'db1.xlsx'))
# data_2 = pd.read_excel(os.path.join(r'C:\Users\GRBH\Desktop', r'db2.xlsx'))
# data_1.to_pickle(os.path.join(
#     r'C:\Users\GRBH\Desktop\GitHub Repositories\coastlib\coastlib\design\design_data',
#     r'aisc_shapes_db_14.1_us.pyc'
# ))
# data_2.to_pickle(os.path.join(
#     r'C:\Users\GRBH\Desktop\GitHub Repositories\coastlib\coastlib\design\design_data',
#     r'aisc_shapes_db_14.1_si.pyc'
# ))


# sections_us = pd.read_pickle(
#     os.path.join(
#         r'C:\Users\GRBH\Desktop\GitHub Repositories\coastlib\coastlib\design\design_data',
#         'aisc_shapes_db_14.1_us.pyc'
#         )
# )
# sections_si = pd.read_pickle(
#     os.path.join(
#         r'C:\Users\GRBH\Desktop\GitHub Repositories\coastlib\coastlib\design\design_data',
#         'aisc_shapes_db_14.1_si.pyc'
#         )
# )


class AiscSection:
    """
    Steel section object which takes section parameters from
    AISC Shapes Database version 14.1 (V14.10)
    AISC Cnstruction Manual, 14th Edition, 3rd printing
    """

    def __init__(self, label, system='us'):

        if system == 'us':
            _sections = sections_us
        elif system == 'si':
            _sections = sections_si
        else:
            raise ValueError(fr'System {system} not recognized. Use us or si')

        self.system = system
        self.label = label.upper()
        if label.upper() in _sections['AISC_Manual_Label'].values and system == 'us':
            _slice = _sections[_sections['AISC_Manual_Label'] == label.upper()]
            self.W = _slice['W'].values[0] * ureg.lb / ureg.ft
            self.A = _slice['A'].values[0] * ureg.inch ** 2
            self.d = _slice['d'].values[0] * ureg.inch
            self.ddet = _slice['ddet'].values[0] * ureg.inch
            self.Ht = _slice['Ht'].values[0] * ureg.inch
            self.h = _slice['h'].values[0] * ureg.inch
            self.OD = _slice['OD'].values[0] * ureg.inch
            self.bf = _slice['bf'].values[0] * ureg.inch
            self.bfdet = _slice['bfdet'].values[0] * ureg.inch
            self.B = _slice['B'].values[0] * ureg.inch
            self.b = _slice['b'].values[0] * ureg.inch
            self.ID = _slice['ID'].values[0] * ureg.inch
            self.tw = _slice['tw'].values[0] * ureg.inch
            self.twdet = _slice['twdet'].values[0] * ureg.inch
            self.tf = _slice['tf'] * ureg.inch

        elif label.upper() in _sections['AISC_Manual_Label'].values and system == 'si':
            _slice = _sections[_sections['AISC_Manual_Label'] == label.upper()]
            self.W = _slice['W'].values[0] * ureg.kg / ureg.m
            self.A = _slice['A'].values[0] * ureg.mm ** 2
            self.d = _slice['d'].values[0] * ureg.mm
            self.ddet = _slice['ddet'].values[0] * ureg.mm

        else:
            raise ValueError(fr'Section {label} not found. Check spelling for AISC Manual Labels')

        self.frame = _slice.T
        self.frame.columns = ['Value']
        self.frame = self.frame[self.frame['Value'] != '–']

    def __repr__(self):

        representation = f'AISC Steel Section\n'
        representation += f'======================================\n\n'
        representation += f'Section label            {self.label}\n'

        try:
            representation += f'Nominal weight - W       {self.W:.2f}\n'
        except ValueError:
            representation += f'Nominal weight - W       -\n'

        try:
            representation += f'Area - A                 {self.A:.2f}\n'
        except ValueError:
            representation += f'Area - A                 -\n'

        try:
            representation += f'Overall depth - d        {self.d:.2f}\n'
        except ValueError:
            representation += f'Overall depth - d        -\n'

        try:
            representation += f'Detailing depth - ddet   {self.ddet:.2f}\n\n'
        except ValueError:
            representation += f'Detailing depth - ddet   -\n\n'

        representation += f'======================================\n'
        representation += f'AISC Construction Manual, 14th Edition'

        return representation

    def convert(self, to='si'):

        if to not in ['si', 'us']:
            raise ValueError(f'System "{to}" not recognized. Use us or si')

        if self.system == 'us':
            if to == 'si':
                self.label = sections_si['AISC_Manual_Label'].values[
                    sections_us['AISC_Manual_Label'] == self.label.upper()
                ][0]
            else:
                return

        elif self.system == 'si':
            if to == 'us':
                self.label = sections_us['AISC_Manual_Label'].values[
                    sections_si['AISC_Manual_Label'] == self.label.upper()
                ][0]
            else:
                return

        self.__init__(label=self.label, system=to)


if __name__ == '__main__':
    test = AiscSection('hss14x4x3/8')
    print(test)
