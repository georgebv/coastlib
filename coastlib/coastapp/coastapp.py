#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shlex
import sys
from coastlib.coastapp.spm_app import main as spm_app


pc_name = os.environ['COMPUTERNAME'].lower()
echo = False
cPATH = os.path.dirname(os.path.abspath(__file__))
costeira_bin = r'C:\Users\GRBH\Desktop\GitHub Repositories\Costeira\costeira\bin'


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


def help():
    print('''
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                                                                           │
   │                     Recongized Coastlib App commands                      │
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
   │cPATH           path to the coastapp root                                  │
   │                                                                           │
   │spm             launches the SPM program                                   │
   │                                                                           │
   │ls              list files in directory                                    │
   │                                                                           │
   │cd              change directory                                           │
   │                                                                           │
   │exit            exits the coastapp program                                 │
   │                                                                           │
   └───────────────────────────────────────────────────────────────────────────┘
    ''')


def command_line(echo=echo, cPATH=cPATH):
    while True:
        # Parse the coastapp line
        coastapp_line = input('\n{pc_name}@coastlib:~$ '.format(pc_name=pc_name))
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
            help()

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

        # Setup the 'cPATH' variable
        elif coastapp_line[0] == 'cPATH':
            if len(coastapp_line) == 2:
                if input('You are about to change cPATH value to "{0}". '
                         'Proceed? (y/n) '.format(coastapp_line[1])) in ['y', 'Y']:
                    cPATH = coastapp_line[1]
            elif len(coastapp_line) == 1:
                print('Coastapp PATH is "{0}"'.format(cPATH))
            else:
                print('ERROR: cPATH command takes exactly one argument')

        elif coastapp_line[0] == 'spm':
            if len(coastapp_line) > 1:
                print('The spm command takes no arguments')
            else:
                spm_app()
                os.system('cls')
                logo()

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
