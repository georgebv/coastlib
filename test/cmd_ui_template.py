#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shlex


pc_name = os.environ['COMPUTERNAME'].lower()
echo = False


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
   │ls              list files in directory                                    │
   │                                                                           │
   │cd              change directory                                           │
   │                                                                           │
   │exit            exits the SMP program                                      │
   │                                                                           │
   └───────────────────────────────────────────────────────────────────────────┘
    ''')


def run_smp(options):
    if 'SMP.INP' not in os.listdir():
        print('\'SMP.INP\' file was not found in the working directory \'{dir}\'.\n'
              'Either change directory, place SMP.INP into \'{dir}\',\n'
              'or run the \'smpinp\' command.'.format(dir=os.getcwd()))
    else:
        os.system('smp93.exe')


def main(echo=echo):
    print('Default echo off')
    while True:
        # Parse the SMP line
        smp_line = input('\033[1m' + '{pc_name}@smp:~$ '.format(pc_name=pc_name) + '\033[0m')
        try:
            smp_line = shlex.split(smp_line, posix=False)
        except ValueError:
            try:
                smp_line = shlex.split(smp_line.replace('\\', '\\\\'), posix=False)
            except ValueError:
                print('Bad syntax in \'{inp}\''.format(inp=smp_line))
        if echo:
            print(smp_line)

        # Execute commands
        if smp_line[0] == 'exit':
            break
            
        elif smp_line[0] == 'echo':
            if len(smp_line) == 2:
                if smp_line[1] == 'on':
                    echo = True
                    print('Echo is on')
                elif smp_line[1] == 'off':
                    echo = False
                    print('Echo is off')
                else:
                    print('Incorrect syntax. Use \'echo on\' or \'echo off\'')
            elif len(smp_line) == 1:
                if echo:
                    print('Echo is on')
                else:
                    print('Echo is off')
            else:
                print('Incorrect syntax. Use \'echo on\' or \'echo off\'')

        elif smp_line[0] == 'help' or smp_line[0] == '-h':
            help()
            
        elif smp_line[0] == 'smp93':
                run_smp(smp_line)

        elif smp_line[0] == 'cd':
            try:
                os.chdir(smp_line[1])
            except FileNotFoundError:
                print('Directory \'{dir}\' does not exist. Please enter a valid UNC path.'.format(dir=smp_line[1]))
            except IndexError:
                print(os.getcwd())

        elif smp_line[0] == 'ls':
            os.system('dir')

        elif smp_line[0] == 'cmd':
            os.system(' '.join(smp_line[1:]))

        else:
            print('Command \'{command}\' not recognized. '
                  'Use \'help\' for a list of available commands.'.format(command=' '.join(smp_line)))


if __name__ == '__main__':
    os.system('color 2')
    os.system('cls')
    print('''
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄       ▄▄  ▄▄▄▄▄▄▄▄▄▄▄               ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
▐░░░░░░░░░░░▌▐░░▌     ▐░░▌▐░░░░░░░░░░░▌             ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
▐░█▀▀▀▀▀▀▀▀▀ ▐░▌░▌   ▐░▐░▌▐░█▀▀▀▀▀▀▀█░▌             ▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀▀▀▀▀▀█░▌
▐░▌          ▐░▌▐░▌ ▐░▌▐░▌▐░▌       ▐░▌             ▐░▌       ▐░▌          ▐░▌
▐░█▄▄▄▄▄▄▄▄▄ ▐░▌ ▐░▐░▌ ▐░▌▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌
▐░░░░░░░░░░░▌▐░▌  ▐░▌  ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀█░▌▐░▌   ▀   ▐░▌▐░█▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀█░▌ ▀▀▀▀▀▀▀▀▀█░▌
          ▐░▌▐░▌       ▐░▌▐░▌                                 ▐░▌          ▐░▌
 ▄▄▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░▌                        ▄▄▄▄▄▄▄▄▄█░▌ ▄▄▄▄▄▄▄▄▄█░▌
▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░▌                       ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
 ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀                         ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀ 
    ''')
    main()
    os.system('color')
    os.system('cls')
