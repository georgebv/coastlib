from coastlib.models.airy import solve_dispersion_relation
import math
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import scipy.stats as sps
import os


######################
# Setup TeX preamble #
######################

preamble = r"""\documentclass[12pt, letterpaper]{article}

\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{graphicx}
\usepackage{transparent}
\usepackage{microtype}
\usepackage[dvipsnames]{xcolor}
\usepackage[left=1in, right=1in, top=1.5in, bottom=1in, headsep=1in]{geometry}
\usepackage[none]{hyphenat}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.2in}
\usepackage{fancyhdr}
\usepackage{mathtools}
\usepackage{multicol}
\usepackage{multirow}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows}
\usepackage{caption}
\usepackage{subcaption}
\usepackage[export]{adjustbox}
\usepackage{pdflscape}
\usepackage{float}

\pagestyle{fancy}
\usepackage{lastpage}
\usepackage{titlesec}
\renewcommand{\baselinestretch}{1.5}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\fancyheadoffset{0.65in}

\begin{document}
"""

title = r"""							% TITLE SECTION
\begin{{multicols}}{{2}}
\hspace*{{-0.1in}}{{\sffamily
	{{\normalsize
		\begin{{tabular}}{{ll}}
		{{\large\textbf{{MEMO}}}}&\\[0.2in]
		{{\color{{RedOrange}}\textbf{{TITLE}}}}& {TITLE} \\
		{{\color{{RedOrange}}\textbf{{DATE}}}}& \today\\
		{{\color{{RedOrange}}\textbf{{TO}}}}& {TO}\\
		{{\color{{RedOrange}}\textbf{{COPY}}}}& {COPY}\\
		{{\color{{RedOrange}}\textbf{{FROM}}}}& {FROM}\\
		{{\color{{RedOrange}}\textbf{{PROJECT NO}}}}& {PROJECT}\\
		\end{{tabular}}
	}}
}}
\hspace*{{1.2in}}{{\sffamily
		{{\normalsize
			\begin{{tabular}}{{rl}}
			\\[0.2in]
			{{\color{{RedOrange}}\textbf{{ADDRESS}}}}& \footnotesize{{COWI North America, Inc.}}\\
			{{}}& \footnotesize{{276 5th Avenue}}\\
			{{}}& \footnotesize{{Suite 1006}}\\
			{{}}& \footnotesize{{New York, NY 10001}}\\
			{{}}& \footnotesize{{USA}}\\%[0.2in]
			{{\color{{RedOrange}}\textbf{{TEL}}}}& \footnotesize{{+1 (646) 545 2125}}\\
			{{\color{{RedOrange}}\textbf{{WWW}}}}& \footnotesize{{cowi-na.com}}\\
			\end{{tabular}}
		}}
	}}
\end{{multicols}}
"""

fheader = r"""								% HEADER
\rhead{
\includegraphics[width=2in]{./Images/COWI_logo.png}\\[0.2in]
{{\sffamily\normalsize\color{RedOrange}\textbf{PAGE}} \thepage/\pageref{LastPage}}
}
\lhead{}
								% FOOTER
{\transparent{0.6}\tikz[overlay,remember picture] \node[opacity=0.3, at=(current page.south east),anchor=south east,inner sep=0.2in] {\includegraphics[width=6in]{./Images/origami_memo.png}};}
\cfoot{}
\pagebreak
"""

##################################
# Setup the body of the document #
##################################

body = r"""							% DOCUMENT BODY
\section{{{SECTION_1}}}
Something very technical goes here\\


We can even have some fancy math!

This is the dispersion relation:
$$\omega^2 = gk\, tanh(kh)$$
where,
$$\omega = \frac{{2\pi}}{{T}} = \frac{{2\cdot 3.14}}{{{VALUE_T}}} = {VALUE_omega}$$
$$k = \frac{{2\pi}}{{L}}$$

Now lets solve it for wave number $k$ with $g=9.81\, m/s^2$ and $h={VALUE_h}\, m$:
$${VALUE_omega}^2=9.81\cdot k\cdot tanh(k\cdot {VALUE_h})$$
Python finds the solution with the iterative Newton-Rhapson method, which gives us:
$$k = {VALUE_k}$$
which, in turn, gives us wave length:
$$L = \frac{{2\pi}}{{k}} = \frac{{2\cdot 3.14}}{{{VALUE_k}}} = {VALUE_L}\, m$$

\pagebreak
We can also have figures automatically generated!
\begin{{figure}}[h]
	\centering
	\includegraphics[height=0.3\textheight]{{./Images/{VALUE_path1}}}
	\caption{{Square root}}
\end{{figure}}

\begin{{figure}}[h]
	\centering
	\includegraphics[height=0.3\textheight]{{./Images/{VALUE_path2}}}
	\caption{{{VALUE_rvals} normally distributed random values}}
\end{{figure}}
"""

############################
# Perform calulations here #
############################

path = os.path.join(os.getcwd(), 'Tex')
document_name = '\PyTeX.tex'

# Disperion relation
T = 15 # T, sec
h = 20 # depth, m
omega = 2 * math.pi / T
L = solve_dispersion_relation(T, h)
k = 2 * math.pi / L

# Square root plot
with plt.style.context('bmh'):
    plt.figure(figsize=(18, 12))
    plt.plot(np.arange(1, 110, 1), np.sqrt(np.arange(1, 110, 1)))
    plt.savefig(path + r'\Images\sqrt.png', dpi=300)
    plt.close()

# Histogram plot
size = 10 ** 4
random_values = sps.norm.rvs(0, size=size)
with plt.style.context('bmh'):
    plt.figure(figsize=(15, 10))
    plt.hist(random_values, bins=50, rwidth=0.95)
    plt.savefig(path + r'\Images\hist.png', dpi=300)
    plt.close()

########################
# Compile the TeX file #
########################

TeX = ''.join(
    [
        preamble,
        title.format(
            TITLE='Python / TeX Introduction',
            TO='Todd Manson',
            COPY='Rebeca Gomez',
            FROM='Georgii Bocharov',
            PROJECT='A123456'
        ),
        fheader,
        body.format(
            SECTION_1='My first PyTeX section',
            VALUE_k=round(k, 2),
            VALUE_omega=round(omega, 2),
            VALUE_T=round(T, 2),
            VALUE_h=round(h, 2),
            VALUE_L=round(L, 2),
            VALUE_path1='sqrt.png',
            VALUE_path2='hist.png',
            VALUE_rvals=size
        ),
        r"""\end{document}"""
    ]
)

with open(path + document_name, 'w') as f:
    f.write(TeX)

cmd = ['pdflatex', '-interaction', 'nonstopmode', path + document_name]
os.chdir(path)
for i in range(2):
    proc = subprocess.Popen(cmd)
    proc.communicate()
