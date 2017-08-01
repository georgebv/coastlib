#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shlex
import sys
from coastlib.coastapp.spm_app import main as spm_app
import coastlib


def logo():
    print(r'''
       ______                     __   __ _  __  
      / ____/____   ____ _ _____ / /_ / /(_)/ /_ 
     / /    / __ \ / __ `// ___// __// // // __ \
    / /___ / /_/ // /_/ /(__  )/ /_ / // // /_/ /
    \____/ \____/ \__,_//____/ \__//_//_//_.___/ 
        ___                                      
       /   |   ____   ____                       
      / /| |  / __ \ / __ \                      
     / ___ | / /_/ // /_/ /                      
    /_/  |_|/ .___// .___/                       
           /_/    /_/                            
        ''')  # slant fitted/fitted


def _help():
    print('''
   ┌────────────────────────────────────────────────────────────────────────────┐
   │                                                                            │
   │                     Recongized Coastlib App Commands                       │
   │                                                                            │
   ├─────────────────────────────┬──────────────────────────────────────────────┤
   │command                      │  action                                      │
   ├─────────────────────────────┼──────────────────────────────────────────────┤
   │                             │                                              │
   │help                         │  provides a list of available commands       │
   │                             │                                              │
   │echo                         │  sets the echo variable on or off.           │
   │                             │  if on, echoes debug info                    │
   │                             │                                              │
   │cmd                          │  sends everything after cmd to command line  │
   │                             │                                              │
   │spm                          │  launches the SPM program                    │
   │                             │                                              │
   │ls                           │  list files in directory                     │
   │                             │                                              │
   │cd                           │  change directory                            │
   │                             │                                              │
   │Fenton -hs [wave height]     │  solve Fenton wave and get a report          │
   │       -tp [wave period]     │                                              │
   │       -d [depth]            │                                              │
   │                             │                                              │
   │Airy -hs [wave height]       │  solve Airy wave and get a report            │
   │     -tp [wave period]       │                                              │
   │     -d [depth]              │                                              │
   │                             │                                              │
   │                             │                                              │
   │exit                         │  exits the coastapp program                  │
   │                             │                                              │
   └─────────────────────────────┴──────────────────────────────────────────────┘
''')


def command_line(echo=False):
    while True:
        # Parse the coastapp line
        coastapp_line = input('\n{pc_name}@coastlib:~$ '.format(pc_name=os.environ['COMPUTERNAME'].lower()))
        try:
            coastapp_line = shlex.split(coastapp_line, posix=False)
        except ValueError:
            try:
                coastapp_line = shlex.split(coastapp_line.replace('\\', '\\\\'), posix=False)
            except ValueError:
                print('Bad syntax in \"{inp}\"'.format(inp=coastapp_line))
        if echo:
            print(coastapp_line)

        # Execute commands
        # Pass if line is empty
        if len(coastapp_line) == 0:
            pass

        # Exit
        elif coastapp_line[0] == 'exit':
            os.system('cls')
            if __name__ == '__main__':
                sys.exit(0)
            else:
                break

        # Setup <echo> variable
        elif coastapp_line[0] == 'echo':
            if len(coastapp_line) == 2:
                if coastapp_line[1] == 'on':
                    echo = True
                    print('Echo is on')
                elif coastapp_line[1] == 'off':
                    echo = False
                    print('Echo is off')
                else:
                    print('Incorrect syntax. Use <echo on> or <echo off>')
            elif len(coastapp_line) == 1:
                if echo:
                    print('Echo is on')
                else:
                    print('Echo is off')
            else:
                print('Incorrect syntax. Use <echo on> or <echo off>')

        # Call help function
        elif coastapp_line[0] == 'help':
            _help()

        # Call Windows 'cd' command
        elif coastapp_line[0] == 'cd':
            try:
                os.chdir(coastapp_line[1])
            except FileNotFoundError:
                print('Directory \"{dir}\" does not exist. '
                      'Please enter a valid UNC path.'.format(dir=coastapp_line[1]))
            except IndexError:
                print(os.getcwd())
            except OSError:
                os.chdir(coastapp_line[1][1:-1])

        # Call 'dir' if Windows or 'ls' if Unix
        elif coastapp_line[0] == 'ls':
            if os.name == 'nt':
                os.system('dir')
            else:
                os.system('ls')

        # Call terminal/command line
        elif coastapp_line[0] == 'cmd':
            if len(coastapp_line) == 1:
                print('Correct usage is <cmd> [input1...] [input2...] ...')
            else:
                os.system(' '.join(coastapp_line[1:]))

        # Run the SPM program
        elif coastapp_line[0] == 'spm':
            if len(coastapp_line) > 1:
                print('The spm command takes no arguments')
            else:
                spm_app()
                os.system('cls')
                logo()

        # Solve a Fenton wave and get a summary report
        elif coastapp_line[0] == 'Fenton':
            try:
                _hs = float(coastapp_line[coastapp_line.index('-hs') + 1])
                _tp = float(coastapp_line[coastapp_line.index('-tp') + 1])
                _d = float(coastapp_line[coastapp_line.index('-d') + 1])
                _fenton_wave = coastlib.FentonWave(wave_height=_hs, wave_period=_tp, depth=_d)
                print('\n' + '=' * 75)
                print(_fenton_wave.report())
                print('=' * 75)
            except Exception as _e:
                print('Bad syntax. Got {}\n'
                      'Correct syntax (any order after <Fenton>) is:\n'
                      '    <Fenton> <-hs> [wave height] <-tp> [wave period] <-d> [depth]'.format(_e))

        # Solve an Airy wave and get a summary report
        elif coastapp_line[0] == 'Airy':
            try:
                _hs = float(coastapp_line[coastapp_line.index('-hs') + 1])
                _tp = float(coastapp_line[coastapp_line.index('-tp') + 1])
                _d = float(coastapp_line[coastapp_line.index('-d') + 1])
                _airy_wave = coastlib.AiryWave(wave_height=_hs, wave_period=_tp, depth=_d)
                print('\n' + '=' * 31)
                print(_airy_wave.as_dataframe())
                print('=' * 31)
            except Exception as _e:
                print('Bad syntax. Got {}\n'
                      'Correct syntax (any order after <Fenton>) is:\n'
                      '    <Airy> <-hs> [wave height] <-tp> [wave period] <-d> [depth]'.format(_e))
            pass

        else:
            print('Command <{command}> not recognized. '
                  'Use <help> for a list of available commands.'.format(command=' '.join(coastapp_line)))


def main():
    os.system('color 2')
    os.system('cls')
    logo()
    command_line()
    os.system('color')
    os.system('cls')


if __name__ == '__main__':
    main()
