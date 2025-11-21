#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define function to scale color
def get_scaled_color(value, color):
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = mcolors.LinearSegmentedColormap.from_list('scale', ['white', color])
    htmlColor = mcolors.rgb2hex(cmap(norm(value)))
    # Remove leading '#'
    return htmlColor[1:]

# Load TSV file
file_path = 'DiscoveryWorld Scenarios - Final Scores for Table.tsv'
#data = pd.read_csv(file_path, sep='\t')
# Load the TSV without Pandas
data = []
with open(file_path, 'r') as file:
    for line in file:
        lineFiltered = line.replace("\n", "")
        data.append(lineFiltered.split('\t'))

# Header
header = data[0]
lines = data[1:]

# Remove any blank lines
filteredLines = []
for line in lines:
    if (len(line[0]) > 0) or (len(line[1]) > 0) or (len(line[2]) > 0):
        filteredLines.append(line)
lines = filteredLines

# Check the column names
print("Columns in the TSV file:", header)

# Define placeholder colors
#colors = ['#FF6347', '#4682B4', '#32CD32']  # Tomato, SteelBlue, LimeGreen
colors = ['#811fb4', '#1f77b4', '#2ca02c']  # Purple, Blue, Green


# Start LaTeX table
latex_table = r"""
%
%   Table: Main Results (Discovery Tasks)
%
\begin{table*}[!t]
\caption{\footnotesize Placeholder }
\label{tab:agent-performance}
\centering
\scriptsize
%\vspace{8mm}
\begin{tabular}{lllccc c ccc c ccc ccc}
\toprule
\multicolumn{3}{c}{~} & \multicolumn{3}{c}{\textbf{ReACT}} & & \multicolumn{3}{c}{\textbf{Plan+Execute}} & & \multicolumn{3}{c}{\textbf{Hypothesizer}}\\
\textbf{\#} & \textbf{Topic} & \textbf{Task} & \rot{Procedure} & \rot{Completion} & \rot{Knowledge} & & \rot{Procedure} & \rot{Completion} & \rot{Knowledge} & &  \rot{Procedure} & \rot{Completion} & \rot{Knowledge} \\
\midrule
"""

# Grouping data by topic
current_topic = None
row_num = 0

def safeConvert(value):
    try:
        return float(value)
    except:
        return -1

def dec2(value):
    return "{:.2f}".format(value)

atAverage = False
for lineNum, line in enumerate(lines):
    print("line: " + str(line))
    if (lineNum % 4 == 0) and (len(line[0]) > 0):
        print("line: " + str(line))
        topic = line[header.index('#')]
        taskDesc = line[header.index('Task')]

        latex_table += r"\multicolumn{2}{l}{\textbf{" + topic + "}} & \multicolumn{11}{l}{" + taskDesc + "} &  \\\\ \n"

    else:
        topic = line[header.index('#')]
        difficulty = line[header.index('Topic')]
        task = line[header.index('Task')]
        react_procedure = line[header.index('ReACT_Procedure')]
        react_completion = line[header.index('ReACT_Completion')]
        react_knowledge = line[header.index('ReACT_Knowledge')]
        plan_execute_procedure = line[header.index('PlanExecute_Procedure')]
        plan_execute_completion = line[header.index('PlanExecute_Completion')]
        plan_execute_knowledge = line[header.index('PlanExecute_Knowledge')]
        hypothesizer_procedure = line[header.index('Hypothesizer_Procedure')]
        hypothesizer_completion = line[header.index('Hypothesizer_Completion')]
        hypothesizer_knowledge = line[header.index('Hypothesizer_Knowledge')]

        react_procedure = safeConvert(react_procedure)
        react_completion = safeConvert(react_completion)
        react_knowledge = safeConvert(react_knowledge)
        plan_execute_procedure = safeConvert(plan_execute_procedure)
        plan_execute_completion = safeConvert(plan_execute_completion)
        plan_execute_knowledge = safeConvert(plan_execute_knowledge)
        hypothesizer_procedure = safeConvert(hypothesizer_procedure)
        hypothesizer_completion = safeConvert(hypothesizer_completion)
        hypothesizer_knowledge = safeConvert(hypothesizer_knowledge)


        if (len(line[0]) > 0):
            row_num = int(topic)
            latex_table += f"{row_num}  & {difficulty} & {task} "
        else:
            if (not atAverage):
                latex_table += """\midrule\n"""
                atAverage = True
            latex_table += """\\rowcolor[HTML]{F3F3F3}\n"""
            latex_table += """\multicolumn{3}{l}{\\textbf{""" + difficulty + """}}"""
        #color_row = 'D3D3D3' if row_num % 2 == 0 else 'FFFFFF'
        #latex_table += r"\rowcolor[HTML]{" + color_row + r"}\n"
        #latex_table += f"{row_num}  & {difficulty} & {task} "

        # Add ReACT scores
        latex_table += f"& \\cellcolor[HTML]{{{get_scaled_color(react_procedure, colors[0])}}}{dec2(react_procedure)} "
        latex_table += f"& \\cellcolor[HTML]{{{get_scaled_color(react_completion, colors[1])}}}{dec2(react_completion)} "
        latex_table += f"& \\cellcolor[HTML]{{{get_scaled_color(react_knowledge, colors[2])}}}{dec2(react_knowledge)} & "

        # Add Plan+Execute scores
        latex_table += f"& \\cellcolor[HTML]{{{get_scaled_color(plan_execute_procedure, colors[0])}}}{dec2(plan_execute_procedure)} "
        latex_table += f"& \\cellcolor[HTML]{{{get_scaled_color(plan_execute_completion, colors[1])}}}{dec2(plan_execute_completion)} "
        latex_table += f"& \\cellcolor[HTML]{{{get_scaled_color(plan_execute_knowledge, colors[2])}}}{dec2(plan_execute_knowledge)} & "

        # Add Hypothesizer scores
        latex_table += f"& \\cellcolor[HTML]{{{get_scaled_color(hypothesizer_procedure, colors[0])}}}{dec2(hypothesizer_procedure)} "
        latex_table += f"& \\cellcolor[HTML]{{{get_scaled_color(hypothesizer_completion, colors[1])}}}{dec2(hypothesizer_completion)} "
        latex_table += f"& \\cellcolor[HTML]{{{get_scaled_color(hypothesizer_knowledge, colors[2])}}}{dec2(hypothesizer_knowledge)} \\\\ "

        if (lineNum % 4 == 3):
            latex_table += r"\hline"

        latex_table += "\n"

        print("Processed")
    lineNum += 1

# \midrule
# \rowcolor[HTML]{F3F3F3}
# \multicolumn{3}{l}{\textbf{Average (Easy)}}     & 0.00 & 0.00 & 0.00 &  & 0.00 & 0.00 & 0.00 &  & 0.00 & 0.00 & 0.00 \\
# \rowcolor[HTML]{F3F3F3}
# \multicolumn{3}{l}{\textbf{Average (Normal)}}   & 0.00 & 0.00 & 0.00 &  & 0.00 & 0.00 & 0.00 &  & 0.00 & 0.00 & 0.00 \\
# \rowcolor[HTML]{F3F3F3}
# \multicolumn{3}{l}{\textbf{Average (Challenge)}}& 0.00 & 0.00 & 0.00 &  & 0.00 & 0.00 & 0.00 &  & 0.00 & 0.00 & 0.00 \\

# Close LaTeX table
latex_table += r"""
\bottomrule
\end{tabular}
%\vspace{-4mm}
\end{table*}
"""

# Write LaTeX table to a file
with open('table.tex', 'w') as file:
    file.write(latex_table)

print("LaTeX table generated and saved as 'table.tex'.")
