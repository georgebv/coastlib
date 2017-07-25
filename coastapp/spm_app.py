#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as xml
import numpy as np
import os
import time
import subprocess
import shutil
import shlex
import pandas as pd


single_output = r'D:\Work folders\desktop projects\7 Kemano\1 Mooring\2 Mooring Analysis\outputs\SPM20.out'
# print(','.join([str(float(i)) for i in np.arange(200, 500, 50)]))
echo = True
pc_name = os.environ['COMPUTERNAME'].lower()


def help():
    print('''
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                                                                           │
   │                          Recongized SMP commands                          │
   │                                                                           │
   ├───────────────────────────────────────────────────────────────────────────┤
   │command         action                                                     │
   ├───────────────────────────────────────────────────────────────────────────┤
   │                                                                           │
   │help, -h        provides a list of available commands                      │
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


xml_input = r'D:\Work folders\desktop projects\7 Kemano\1 Mooring\2 Mooring Analysis\Kemano.xml'
inpath = r'D:\Work folders\desktop projects\7 Kemano\1 Mooring\2 Mooring Analysis\inputs'
outpath = r'D:\Work folders\desktop projects\7 Kemano\1 Mooring\2 Mooring Analysis\outputs'
spm_path = r'C:\Users\GRBH\Desktop\GitHub Repositories\Costeira\costeira\bin\SPM.exe'


def generate_input(xml_input, outpath, echo):
    with open(xml_input, 'r') as f:
        tree = xml.parse(f)
        root = tree.getroot()

    def convert(address, root=root, type=float):
        loc_data = root.find(address).text
        try:
            loc_data = loc_data.split(sep=',')
            loc_data = np.array([type(i) for i in loc_data])
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
    SYN = SYN.strip() # A=AmSteel-Blue, S=steel, U=user defined) (-)

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
    NLINESs = convert('./Hawser/Number', type=int)
    PMAXs = convert('./General/MaximumLoad')  # breaking strength of weakest chain (lb)
    PLDs = convert('./Hawser/Preload')  # horizontal pre-load (lb)
    ONDECKs = convert('./Hawser/OnDeckLength')  # on deck hawser length (ft)
    AREA = 0.0  # area of steel (sq. in.)
    ESTRAN = 0.0  # modulus of elasticity of steel line (PSI)

    # Generate input ASCII files
    def gen_loc_inp(CWII, CLI, CWI, WS, SB, PMAX, PLD, ONDECK, LRW, WD, NLINES, MBL):
        s = ['''Analysis Date: 2017-07-21
Project: Kemano
Description: Buoy mooring analysis
Analyst: GRBH
Configuration: Low tide''']
        s += ['{:10d}{:10.2f}{:10.1f}{:5d}{:5d}{:5d}{:5d}{:5d}'.format(NT, TS, ZW, JPLOT, JLIST, JDATA, JPLTLD, NUSER)]
        s += ['    {:1s}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}'
              '    {:1s}'.format(NBUOY, A, B, XD, YD, ANG, LRW, SYN)]
        s += ['{:5d}{:10.2f}'.format(NLINES, MBL)]
        s += ['        {:2s}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}'.format(BUOY, CLII, CWII, CLI, CWI, WS)]
        s += [
            '{:10.1f}{:10.1f}{:10.0f}{:10.0f}{:10.0f}{:10.2f}{:10.0f}'.format(SB, WD, PMAX, PLD, ONDECK, AREA, ESTRAN)]
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
        with open( r'C:\SPMinp\SPM{ind:d}.in'.format(ind=i), 'w') as outfile:
            outfile.write(inputs[i])


def run_spm(spm_path, inpath, outpath, echo, wait_time=1):
    if not os.path.exists(r'C:\SPMout'):
        os.makedirs(r'C:\SPMout')
    for file in os.listdir(inpath):
        command = '\"' + spm_path + '\" \"C:\SPMinp' + '\\' + file + '\" \"C:\SPMout' + '\\' + file[:-2] + 'out\"'
        p = subprocess.call(command)
        if echo:
            print('Called in cmd - "{call}"'.format(call=command))
        time.sleep(wait_time)
    shutil.rmtree(r'C:\SPMinp')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    try:
        shutil.copytree(r'C:\SPMout', outpath)
    except FileExistsError:
        for file in os.listdir(r'C:\SPMout'):
            shutil.copyfile(r'C:\SPMout\\'+file, outpath + '\\' + file)
    shutil.rmtree(r'C:\SPMout')
    if echo:
        print()


def parse_output(outpath, echo, excel=False, csv=False):
    def parse_single(single_output):
        with open(single_output, 'r') as file:
            raw_data = file.readlines()
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
        data.insert(0, [single_output.split(sep='\\')[-1].split(sep='.')[0][3:] for i in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[55].split(sep=' ')))[4]) for i in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[55].split(sep=' ')))[5]) for i in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[55].split(sep=' ')))[6]) for i in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[55].split(sep=' ')))[7]) for i in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[55].split(sep=' ')))[8]) for i in range(len(data[0]))])
        data.append([float(list(filter(None, raw_data[65].split(sep=' ')))[3]) for i in range(len(data[0]))])
        data = np.array(data)
        fl = np.vectorize(lambda x: float(x))
        data = fl(data)
        data = pd.DataFrame(data=data.T, columns=columns)
        col = data.columns
        data['Chain length / Water depth'] = (data[col[14]].values + data[col[15]].values) / (data[col[19]].values)
        return data

    outputs = []
    for file in os.listdir(outpath):
        if file.endswith('.out'):
            outputs += [parse_single(single_output=outpath + '\\' + file)]
            if echo:
                print('Parsed "{0}"'.format(outpath + '\\' + file))
    output = pd.concat(outputs)
    if excel:
        output.to_excel(outpath + '\Data.xlsx')
        print('Saved output to "{0}"'.format(outpath + '\Data.xlsx'))
    if csv:
        output.to_csv(outpath + '\Data.csv')
        print('Saved output to "{0}"'.format(outpath + '\Data.csv'))
    output.to_pickle(outpath + '\Data.pyc')
    print('Saved output to "{0}"'.format(outpath + '\Data.pyc'))


def main(echo=echo, xml_input=xml_input, inpath=inpath, outpath=outpath, spmpath=spm_path):
    while True:
        # Parse the SMP line
        spm_line = input('{pc_name}@spm:~$ '.format(pc_name=pc_name))
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
        if spm_line[0] == 'exit':
            break

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

        elif spm_line[0] == 'help' or spm_line[0] == '-h':
            help()

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
                if len(spm_line) > 1:
                    try:
                        run_spm(spm_path=spmpath, inpath=inpath, outpath=outpath,
                                echo=echo, wait_time=int(spm_line[1]))
                    except ValueError:
                        print('Wait time should be an integer. "{0}" was given instead'.format(spm_line[1]))
                else:
                    run_spm(spm_path=spm_path, inpath=inpath, outpath=outpath, echo=echo, wait_time=1)
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
                print('Directory \'{dir}\' does not exist. Please enter a valid UNC path.'.format(dir=smp_line[1]))
            except IndexError:
                print(os.getcwd())
            except OSError:
                os.chdir(spm_line[1][1:-1])

        elif spm_line[0] == 'ls':
            os.system('dir')

        elif spm_line[0] == 'cmd':
            os.system(' '.join(spm_line[1:]))

        else:
            print('Command \'{command}\' not recognized. '
                  'Use \'help\' for a list of available commands.'.format(command=' '.join(spm_line)))


if __name__ == '__main__':
    os.system('color 2')
    os.system('cls')
    print('''
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄       ▄▄ 
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░▌     ▐░░▌
▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌░▌   ▐░▐░▌
▐░▌          ▐░▌       ▐░▌▐░▌▐░▌ ▐░▌▐░▌
▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌ ▐░▐░▌ ▐░▌
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌  ▐░▌  ▐░▌
 ▀▀▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░▌   ▀   ▐░▌
          ▐░▌▐░▌          ▐░▌       ▐░▌
 ▄▄▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌       ▐░▌
▐░░░░░░░░░░░▌▐░▌          ▐░▌       ▐░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀            ▀         ▀ 
    ''')
    main()
    os.system('color')
    os.system('cls')
