raw_tex = '''\documentclass[12pt, letterpaper]{{article}}\n
\\author{{Georgii Bocharov}}
\date{{\\today}}
\\title{{Wave Load Calculation}}\n
\\usepackage[utf8]{{inputenc}}
\\usepackage[english]{{babel}}
\\usepackage{{graphicx}}
\\usepackage{{transparent}}
\\usepackage{{microtype}}
\\usepackage[dvipsnames]{{xcolor}}
\\usepackage[left=1in, right=1in, top=1.5in, bottom=1in, headsep=1in]{{geometry}}
\\usepackage[none]{{hyphenat}}
\setlength{{\parindent}}{{0pt}}
\\usepackage{{fancyhdr}}
\\usepackage{{mathtools}}\n
\pagestyle{{fancy}}
\\usepackage{{lastpage}}
\\usepackage{{titlesec}}
\\renewcommand{{\\baselinestretch}}{{1.5}}
\\renewcommand{{\headrulewidth}}{{0pt}}
\\renewcommand{{\\footrulewidth}}{{0pt}}
\\fancyheadoffset{{0.65in}}\n
\makeatletter
% we use \prefix@<level> only if it is defined
\\renewcommand{{\@seccntformat}}[1]{{%
  \ifcsname prefix@#1\endcsname
    \csname prefix@#1\endcsname
  \else
    \csname the#1\endcsname\quad
  \\fi}}
% define \prefix@section
\\newcommand\prefix@section{{Calculation case \\thesection: }}
\makeatother\n
\\begin{{document}}
\\rhead{{
\includegraphics[width=2in]{{COWI_logo.png}}\\\[0.2in]
{{
{{\sffamily\\normalsize\color{{RedOrange}}\\textbf{{PAGE}}}} \\thepage/\pageref{{LastPage}}}}
}}
\cfoot{{}}
SAMPLE TEXT
\end{{document}}'''
page = raw_tex.format()

with open('cover.tex', 'w') as f:
    f.write(page)
