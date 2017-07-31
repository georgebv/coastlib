#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as Xml
import numpy as np
import os
import subprocess
import shutil
import shlex
import pandas as pd
import sys
import datetime


def logo():
    print(r'''
       _____  _                __                  
      / ___/ (_)____   ____ _ / /___               
      \__ \ / // __ \ / __ `// // _ \              
     ___/ // // / / // /_/ // //  __/              
    /____//_//_/ /_/ \__, //_/ \___/               
                    /____/                         
        ____          _         __                 
       / __ \ ____   (_)____   / /_                
      / /_/ // __ \ / // __ \ / __/                
     / ____// /_/ // // / / // /_                  
    /_/     \____//_//_/ /_/ \__/                  
        __  ___                     _              
       /  |/  /____   ____   _____ (_)____   ____ _
      / /|_/ // __ \ / __ \ / ___// // __ \ / __ `/
     / /  / // /_/ // /_/ // /   / // / / // /_/ / 
    /_/  /_/ \____/ \____//_/   /_//_/ /_/ \__, /  
                                          /____/   
''')  # slant fitted/fitted


def _help():
    print('''
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                                                                           │
   │                          Recongized SMP commands                          │
   │                                                                           │
   ├───────────────────────────────────────────────────────────────────────────┤
   │command         action                                                     │
   ├───────────────────────────────────────────────────────────────────────────┤
   │                                                                           │
   │help            provides a list of available commands                      │
   │                                                                           │
   │echo            Sets the echo variable on or off. If on, echoes debug info │
   │                                                                           │
   │cmd             sends everything after cmd to command line                 │
   │                                                                           │
   │xml             set xml path                                               │
   │                                                                           │
   │inpath          set input folder path                                      │   
   │                                                                           │
   │outpath         set output folder path                                     │
   │                                                                           │
   │spmpath         set spm.exe path                                           │
   │                                                                           │
   │run             run spm and generate output in outpath                     │
   │                                                                           │
   │parse           parses output to a single xcel table for further processing│
   │                -csv for csv output, -excel for excel output               │
   │                                                                           │
   │ls              list files in directory                                    │
   │                                                                           │
   │cd              change directory                                           │
   │                                                                           │
   │exit            exits the SMP program                                      │
   │                                                                           │
   └───────────────────────────────────────────────────────────────────────────┘
    ''')


# Defaults
_xml_input = r'D:\Work folders\desktop projects\7 Kemano\1 Mooring\2 Mooring Analysis\Kemano.xml'
_inpath = r'D:\Work folders\desktop projects\7 Kemano\1 Mooring\2 Mooring Analysis\inputs'
_outpath = r'D:\Work folders\desktop projects\7 Kemano\1 Mooring\2 Mooring Analysis\outputs'
_spm_path = r'C:\Users\GRBH\Desktop\GitHub Repositories\Costeira\costeira\bin\SPM.exe'


def generate_input(xml_input, outpath, echo):
    with open(xml_input, 'r') as f:
        tree = Xml.parse(f)
        root = tree.getroot()

    def convert(address, stype=float):
        loc_data = root.find(address).text
        try:
            loc_data = loc_data.split(sep=',')
            loc_data = np.array([stype(value) for value in loc_data])
            return loc_data
        except ValueError:
            print('ERROR: Ivalid input in "{add}"! Revise the xml file before proceding'.format(add=address))

    # Inputs
    # These are not used
    NT = 2000  # number of steps in force time history (-)
    TS = 0.96  # time step in seconds (s)
    ZW = 0.0  # Z coordinate of water surface (ft)

    # Constants
    JPLOT = 1  # plot flag (0=no, 1=yes) (-)
    JLIST = 1  # list of results (0=no, 1=yes) (-)
    JDATA = 1  # list program options selected (0=no, 1=yes) (-)
    JPLTLD = 1  # plot load-deflection curve flag (0=no, 1=yes)

    NUSER = 0  # number of user defined load-deflection curves (-)
    NBUOY = "A"  # anchor position option (A = angle) (-)
    A = 0.0  # X chock position (ft)
    B = 0.0  # Y chock position (ft)
    XD = 0.0  # X anchor position; not required if NBUOY = A (ft)
    YD = 0.0  # Y anchor position; not required if NBUOY = A (ft)
    ANG = 0.0  # anchor angle; use 0 for single point mooring (deg clockwise)
    LRWs = convert('./Hawser/HorizontalLength')  # horizontal length chock-to-buoy (ft)
    SYN = root.find('./Hawser/Material').text  # synthetic material type (N=nylon,P=polypropylene,p=old poly.,
    SYN = SYN.strip()  # A=AmSteel-Blue, S=steel, U=user defined) (-)

    BUOY = "DC"  # anchor leg type (DC = double catenary) (-)
    CLIIs = convert('./UpperChain/Length')  # chain length of upper segment (ft)

    CWIIs = convert('./UpperChain/UnitWeight')  # chain unit weight of upper segment (lb/ft)
    CLIs = convert('./LowerChain/Length')  # chain length of lower segment (ft)
    CWIs = convert('./LowerChain/UnitWeight')  # chain unit weight of lower segment (lb/ft)
    WSs = convert('./Sinker/SubmergedWeight')  # submerged weight of sinker (lb)

    SBs = convert('./Hawser/ChockHeight')  # chock height above water line (ft)
    ChartedDepth = convert('./Environment/ChartedDepth')
    WaterLevel = convert('./Environment/WaterLevel')
    WDs = ChartedDepth + WaterLevel  # water depth at anchor (ft)
    MBLs = convert('./Hawser/MBL')
    NLINESs = convert('./Hawser/Number', stype=int)
    PMAXs = convert('./General/MaximumLoad')  # breaking strength of weakest chain (lb)
    PLDs = convert('./Hawser/Preload')  # horizontal pre-load (lb)
    ONDECKs = convert('./Hawser/OnDeckLength')  # on deck hawser length (ft)
    AREA = 0.0  # area of steel (sq. in.)
    ESTRAN = 0.0  # modulus of elasticity of steel line (PSI)

    # Generate input ASCII files
    def gen_loc_inp(cwii, cli, cwi, ws, sb, pmax, pld, ondeck, lrw, wd, nlines, mbl):
        s = ['''Analysis Date: {:%D}
Project: SPM
Description: Mooring analysis
Analyst: GRBH
Configuration: Custom'''.format(datetime.datetime.today())]
        s += ['{:10d}{:10.2f}{:10.1f}{:5d}{:5d}{:5d}{:5d}{:5d}'.format(NT, TS, ZW, JPLOT, JLIST, JDATA, JPLTLD, NUSER)]
        s += ['    {:1s}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}'
              '    {:1s}'.format(NBUOY, A, B, XD, YD, ANG, lrw, SYN)]
        s += ['{:5d}{:10.2f}'.format(nlines, mbl)]
        s += ['        {:2s}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}'.format(BUOY, CLII, cwii, cli, cwi, ws)]
        s += [
            '{:10.1f}{:10.1f}{:10.0f}{:10.0f}{:10.0f}{:10.2f}{:10.0f}'.format(sb, wd, pmax, pld, ondeck, AREA, ESTRAN)]
        return '\n'.join(s)

    inputs = []
    for LRW in LRWs:
        for NLINES in NLINESs:
            for MBL in MBLs:
                for CLII in CLIIs:
                    for CWII in CWIIs:
                        for CLI in CLIs:
                            for CWI in CWIs:
                                for WS in WSs:
                                    for SB in SBs:
                                        for PMAX in PMAXs:
                                            for PLD in PLDs:
                                                for ONDECK in ONDECKs:
                                                    for WD in WDs:
                                                        inputs += [gen_loc_inp(
                                                            CWII, CLI, CWI, WS, SB, PMAX, PLD, ONDECK,
                                                            LRW, WD, NLINES, MBL
                                                        )]
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for i in range(len(inputs)):
        with open(outpath + r'\SPM{ind:d}.in'.format(ind=i), 'w') as outfile:
            outfile.write(inputs[i])
            if echo:
                print('Generated input file "{inp}"'.format(
                    inp=outpath + r'\SPM{ind:d}.in'.format(ind=i)
                ))
        if not os.path.exists(r'C:\SPMinp'):
            os.makedirs(r'C:\SPMinp')
        with open(r'C:\SPMinp\SPM{ind:d}.in'.format(ind=i), 'w') as outfile:
            outfile.write(inputs[i])


def run_spm(spm_path, inpath, outpath, echo):
    temp_inpath = os.path.join(os.environ['ALLUSERSPROFILE'], 'spmtmpin')
    if not os.path.exists(temp_inpath):
        os.makedirs(temp_inpath)

    temp_outpath = os.path.join(os.environ['ALLUSERSPROFILE'], 'spmtmpout')
    if not os.path.exists(temp_outpath):
        os.makedirs(temp_outpath)

    for file in os.listdir(inpath):
        command = '\"' + spm_path + '\"' +\
                  '\"' + temp_inpath + '\\' + file + '\"' +\
                  '\"' + temp_outpath + '\\' + file[:-2] + 'out' + '\"'
        p = subprocess.Popen(command, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        log = []
        while p.poll() is None:
            line = p.stdout.readline()
            try:
                line = line.decode()
                if len(line) > 0:
                    log += [line]
            except AttributeError:
                pass
        if echo:
            print('Called in cmd - "{call}"\n'
                  'Process logs\n'.format(call=command) + '='*75 + '\n')
            print('\n'.join(log))

    shutil.rmtree(temp_inpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    try:
        shutil.copytree(temp_outpath, outpath)
    except FileExistsError:
        for file in os.listdir(temp_outpath):
            shutil.copyfile(os.path.join(temp_outpath, file), os.path.join(outpath, file))
    shutil.rmtree(temp_outpath)
    if echo:
        print()


def parse_output(outpath, echo, excel=False, csv=False):

    def parse_single(single_output):
        with open(single_output, 'r') as _file:
            raw_data = _file.readlines()
        columns = [
            'Test id',
            'Horizontal chain force at buoy (lbs)',
            'Vertical chain force at buoy (lbs)',
            'Resultant chain force at buoy (lbs)',
            'Horizontal distance from anchor to buoy (feet)',
            'Length of upper chain lifted (feet)',
            'Distance sinker is off bottom (feet)',
            'Length of lower chain lifted (feet)',
            'Angle of chain at the anchor (degrees)',
            'Hawser angle (degrees)',
            'Virtual angle (degrees)',
            'Horizontal distance from anchor to chock (feet)',
            'Straight line dist anchor to chock (m)',
            'Virtual tension anchor to chock (N)',
            'Length of upper chain (feet)',
            'Submerged unit weight of upper chain (lbs/ft)',
            'Length of lower chain (feet)',
            'Submerged unit weight of lower chain (lbs/ft)',
            'Submerged weight of sinker (lbs)',
            'Water depth at anchor (feet)',

        ]
        end = raw_data.index('                              LOAD DEFLECTION CURVE IS RISING VERTICALLY.'
                             '  LINEAR EXTRAPOLATION IS NOW IN PROGRESS.\n')
        data = raw_data[82:end-1]
        data = [i.split(sep=' ') for i in data]
        data = [list(filter(None, i)) for i in data]
        data = np.array([[float(value) for value in row] for row in data])
        data = [[data[i][j] for i in range(len(data))] for j in range(len(data[0]))]
        data.insert(0, [single_output.split(sep='\\')[-1].split(sep='.')[0][3:] for _ in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[55].split(sep=' ')))[4]) for _ in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[55].split(sep=' ')))[5]) for _ in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[55].split(sep=' ')))[6]) for _ in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[55].split(sep=' ')))[7]) for _ in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[55].split(sep=' ')))[8]) for _ in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[65].split(sep=' ')))[3]) for _ in range(len(data[0]))])
        data = np.array(data)
        fl = np.vectorize(lambda x: float(x))
        data = fl(data)
        data = pd.DataFrame(data=data.T, columns=columns)
        col = data.columns
        data['Chain length / Water depth'] = (data[col[14]].values + data[col[15]].values) / data[col[19]].values
        return data

    outputs = []
    for file in os.listdir(outpath):
        if file.endswith('.out'):
            outputs += [parse_single(single_output=os.path.join(outpath, file))]
            if echo:
                print('Parsed "{0}"'.format(os.path.join(outpath, file)))
    output = pd.concat(outputs)

    if excel:
        output.to_excel(outpath + '\Data.xlsx')
        print('Saved output to "{0}"'.format(os.path.join(outpath, 'Data.xlsx')))
    if csv:
        output.to_csv(outpath + '\Data.csv')
        print('Saved output to "{0}"'.format(os.path.join(outpath, 'Data.csv')))
    output.to_pickle(outpath + '\Data.pyc')
    print('Saved output to "{0}"'.format(os.path.join(outpath, 'Data.pyc')))


def command_line(echo=False, xml_input=_xml_input, inpath=_inpath, outpath=_outpath, spmpath=_spm_path):
    while True:
        # Parse the SMP line
        spm_line = input('\n{pc_name}@spm:~$ '.format(pc_name=os.environ['COMPUTERNAME'].lower()))
        try:
            spm_line = shlex.split(spm_line, posix=False)
        except ValueError:
            try:
                spm_line = shlex.split(spm_line.replace('\\', '\\\\'), posix=False)
            except ValueError:
                print('Bad syntax in \'{inp}\''.format(inp=spm_line))
        if echo:
            print(spm_line)

        # Execute commands
        # Pass if line is empty
        if len(spm_line) == 0:
            pass

        # Exit
        elif spm_line[0] == 'exit':
            os.system('cls')
            if __name__ == '__main__':
                sys.exit(0)
            else:
                break

        # Setup <echo> variable
        elif spm_line[0] == 'echo':
            if len(spm_line) == 2:
                if spm_line[1] == 'on':
                    echo = True
                    print('Echo is on')
                elif spm_line[1] == 'off':
                    echo = False
                    print('Echo is off')
                else:
                    print('Incorrect syntax. Use \'echo on\' or \'echo off\'')
            elif len(spm_line) == 1:
                if echo:
                    print('Echo is on')
                else:
                    print('Echo is off')
            else:
                print('Incorrect syntax. Use \'echo on\' or \'echo off\'')

        # Call help function
        elif spm_line[0] == 'help':
            _help()

        elif spm_line[0] == 'xml':
            if len(spm_line) > 1:
                xml_input = spm_line[1]
            else:
                print(xml_input)

        elif spm_line[0] == 'inpath':
            if len(spm_line) > 1:
                inpath = spm_line[1]
            else:
                print(inpath)

        elif spm_line[0] == 'outpath':
            if len(spm_line) > 1:
                outpath = spm_line[1]
            else:
                print(outpath)

        elif spm_line[0] == 'spmpath':
            if len(spm_line) > 1:
                spmpath = spm_line[1]
            else:
                print(spmpath)

        elif spm_line[0] == 'run':
            try:
                generate_input(xml_input=xml_input, outpath=inpath, echo=echo)
                run_spm(spm_path=spmpath, inpath=inpath, outpath=outpath, echo=echo)
            except FileNotFoundError:
                print('File "{0}" does not exist'.format(xml_input))

        elif spm_line[0] == 'parse':
            try:
                if '-excel' in spm_line:
                    excel = True
                else:
                    excel = False
                if '-csv' in spm_line:
                    csv = True
                else:
                    csv = False
                parse_output(outpath=outpath, echo=echo, excel=excel, csv=csv)
            except FileNotFoundError:
                print('Nothing to parse in "{0}"'.format(outpath))

        elif spm_line[0] == 'cd':
            try:
                os.chdir(spm_line[1])
            except FileNotFoundError:
                print('Directory \"{dir}\" does not exist. '
                      'Please enter a valid UNC path.'.format(dir=spm_line[1]))
            except IndexError:
                print(os.getcwd())
            except OSError:
                os.chdir(spm_line[1][1:-1])

        # Call 'dir' if Windows or 'ls' if Unix
        elif spm_line[0] == 'ls':
            if os.name == 'nt':
                os.system('dir')
            else:
                os.system('ls')

        # Call terminal/command line
        elif spm_line[0] == 'cmd':
            if len(spm_line) == 1:
                print('Correct usage is <cmd> [input1...] [input2...] ...')
            else:
                os.system(' '.join(spm_line[1:]))

        else:
            print('Command <{command}> not recognized. '
                  'Use <help> for a list of available commands.'.format(command=' '.join(spm_line)))


def main():
    os.system('color 2')
    os.system('cls')
    logo()
    command_line()
    os.system('color')
    os.system('cls')


if __name__ == '__main__':
    main()
