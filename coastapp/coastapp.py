#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shlex

pc_name = os.environ['COMPUTERNAME'].lower()
echo = False
cPATH = r'C:\Users\GRBH\Desktop\GitHub Repositories\coastlib\coastapp'


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
   │help, -h        provides a list of available commands                      │
   │                                                                           │
   │echo            Sets the echo variable on or off. If on, echoes debug info │
   │                                                                           │
   │cmd             sends everything after cmd to command line                 │
   │                                                                           │
   │cPATH           path to the coastapp root                                  │
   │                                                                           │
   │ls              list files in directory                                    │
   │                                                                           │
   │cd              change directory                                           │
   │                                                                           │
   │exit            exits the coastapp program                                 │
   │                                                                           │
   └───────────────────────────────────────────────────────────────────────────┘
    ''')




def main(echo=echo, cPATH=cPATH):
    while True:
        # Parse the coastapp line
        coastapp_line = input('{pc_name}@coastlib:~$ '.format(pc_name=pc_name))
        try:
            coastapp_line = shlex.split(coastapp_line, posix=False)
        except ValueError:
            try:
                coastapp_line = shlex.split(coastapp_line.replace('\\', '\\\\'), posix=False)
            except ValueError:
                print('Bad syntax in \'{inp}\''.format(inp=coastapp_line))
        if echo:
            print(coastapp_line)

        # Execute commands
        if coastapp_line[0] == 'exit':
            break

        elif coastapp_line[0] == 'echo':
            if len(coastapp_line) == 2:
                if coastapp_line[1] == 'on':
                    echo = True
                    print('Echo is on')
                elif coastapp_line[1] == 'off':
                    echo = False
                    print('Echo is off')
                else:
                    print('Incorrect syntax. Use \'echo on\' or \'echo off\'')
            elif len(coastapp_line) == 1:
                if echo:
                    print('Echo is on')
                else:
                    print('Echo is off')
            else:
                print('Incorrect syntax. Use \'echo on\' or \'echo off\'')

        elif coastapp_line[0] == 'help' or coastapp_line[0] == '-h':
            help()

        elif coastapp_line[0] == 'cd':
            try:
                os.chdir(coastapp_line[1])
            except FileNotFoundError:
                print('Directory \'{dir}\' does not exist. Please enter a valid UNC path.'.format(dir=coastapp_line[1]))
            except IndexError:
                print(os.getcwd())
            except OSError:
                os.chdir(coastapp_line[1][1:-1])

        elif coastapp_line[0] == 'ls':
            os.system('dir')

        elif coastapp_line[0] == 'cmd':
            os.system(' '.join(coastapp_line[1:]))

        elif coastapp_line[0] == 'cPATH':
            if len(coastapp_line) == 2:
                cPATH = coastapp_line[1]
            elif len(coastapp_line) == 1:
                print('Coastapp PATH is "{0}"'.format(cPATH))
            else:
                print('ERROR: cPATH command takes exactly one argument')

        elif coastapp_line[0] == 'spm':
            if len(coastapp_line) > 1:
                print('The spm command takes no arguments')
            else:
                os.system(' '.join([
                    'python', '"'+os.path.join(cPATH, 'spm_app.py')+'"',
                ]))

        else:
            print('Command \'{command}\' not recognized. '
                  'Use \'help\' for a list of available commands.'.format(command=' '.join(coastapp_line)))


if __name__ == '__main__':
    os.system('color 2')
    os.system('cls')
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
    ''')
    main()
    os.system('color')
    os.system('cls')
