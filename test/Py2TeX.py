from coastlib.models.linear_wave_theory import solve_dispersion_relation
import math
import matplotlib.pyplot as plt
import numpy as np


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

Now lets solve it for wave number $k$ with $g=9.81m/s^2$ and $h={VALUE_h}m$:
$${VALUE_omega}^2=9.81\cdot k\cdot tanh(k\cdot {VALUE_h})$$
Python finds the solution with the iterative Newton-Rhapson method, which gives us:
$$k = {VALUE_k}$$
which, in turn, gives us wave length:
$$L = \frac{{2\pi}}{{k}} = \frac{{2\cdot 3.14}}{{{VALUE_k}}} = {VALUE_L}m$$

\pagebreak
We can also have figures automatically generated!
\begin{{figure}}[h]
	\centering
	\includegraphics[width=\textwidth]{{./Images/{VALUE_path}}}
	\caption{{Squre root plot}}
\end{{figure}}
"""

############################
# Perform calulations here #
############################

Images = r'.\TeX\Images'

T = 10 # T, sec
h = 5 # depth, m

omega = 2 * math.pi / T
L = solve_dispersion_relation(T, h)
k = 2 * math.pi / L
with plt.style.context('bmh'):
    plt.figure(figsize=(18, 12))
    plt.plot(np.arange(1, 11, 1), np.sqrt(np.arange(1, 11, 1)))
    plt.savefig(Images + r'\image.png', dpi=300)

########################
# Compile the TeX file #
########################

TeX = ''.join(
    [
        preamble,
        title.format(
            TITLE='Python / TeX Introduction',
            TO='John Doe',
            COPY='Jane Doe',
            FROM='Me',
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
            VALUE_path=r'image.png'
        ),
        r"""\end{document}"""
    ]
)

with open(r'.\TeX\test.tex', 'w') as f:
    f.write(TeX)
