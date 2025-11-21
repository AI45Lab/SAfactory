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
#file_path = 'DiscoveryWorld Scenarios - Final Scores for Table.tsv'
#file_path = "paper-results/human/analysisHumanPerformanceFollowOnAverages-withmanualknowledge.tsv"
file_path = "paper-results/human/corrected-rocket-science/analysisHumanPerformanceFollowOnAverages-withmanualknowledge-rocketsciencecorrected.tsv"
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

#Keys for LUT:
# archaeology dating simple_unknown
# archaeology dating_challenge
# combinatorial chemistry_challenge
# combinatorial chemistry_normal
# lost in translation hard_unknown
# lost in translation_normal
# not rocket science_challenge
# not rocket science_normal
# plant nutrients_challenge
# plant nutrients_unknown
# proteomics_challenge
# proteomics_normal
# reactor lab_challenge
# reactor lab_unknown
# space sick_challenge
# space sick_normal

LUT = {
    "archaeology dating simple_unknown": {"topic": "Archaeology", "difficulty": "Normal", "taskDesc": "", "taskID": 8},
    "archaeology dating_challenge": {"topic": "Archaeology", "difficulty": "Challenge", "taskDesc": "", "taskID": 9},
    "combinatorial chemistry_challenge": {"topic": "Chemistry", "difficulty": "Challenge", "taskDesc": "", "taskID": 6},
    "combinatorial chemistry_normal": {"topic": "Chemistry", "difficulty": "Normal", "taskDesc": "", "taskID": 5},
    "lost in translation hard_unknown": {"topic": "Translation", "difficulty": "Challenge", "taskDesc": "", "taskID": 24},
    "lost in translation_normal": {"topic": "Translation", "difficulty": "Normal", "taskDesc": "", "taskID": 23},
    "not rocket science_challenge": {"topic": "Rocket Science", "difficulty": "Challenge", "taskDesc": "", "taskID": 21},
    "not rocket science_normal": {"topic": "Rocket Science", "difficulty": "Normal", "taskDesc": "", "taskID": 20},
    "plant nutrients_challenge": {"topic": "Plant Nutrients", "difficulty": "Challenge", "taskDesc": "", "taskID": 15},
    "plant nutrients_unknown": {"topic": "Plant Nutrients", "difficulty": "Normal", "taskDesc": "", "taskID": 14},
    "proteomics_challenge": {"topic": "Proteomics", "difficulty": "Challenge", "taskDesc": "", "taskID": 3},
    "proteomics_normal": {"topic": "Proteomics", "difficulty": "Normal", "taskDesc": "", "taskID": 2},
    "reactor lab_challenge": {"topic": "Reactor Lab", "difficulty": "Challenge", "taskDesc": "", "taskID": 12},
    "reactor lab_unknown": {"topic": "Reactor Lab", "difficulty": "Normal", "taskDesc": "", "taskID": 11},
    "space sick_challenge": {"topic": "Space Sick", "difficulty": "Challenge", "taskDesc": "", "taskID": 18},
    "space sick_normal": {"topic": "Space Sick", "difficulty": "Normal", "taskDesc": "", "taskID": 17},
}

# Start LaTeX table
latex_table = r"""
%
%   Table: Human Performance (Discovery Tasks)
%
\begin{table*}[!t]
\caption{\footnotesize Placeholder }
\label{tab:agent-performance}
\centering
\scriptsize
%\vspace{8mm}
\begin{tabular}{lllccc c cccccc}
\toprule
\multicolumn{3}{c}{~} & \multicolumn{7}{c}{\textbf{Human Performance}} \\
\textbf{\#} & \textbf{Topic} & \textbf{Task} & \rot{Procedural} & \rot{Completion} & \rot{Knowledge} & & \rot{Avg. Steps} & \rot{Movement Steps} & \rot{Action Steps} & \rot{Prop. Action Steps} & & \rot{numSamples}\\
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
linesToExport = []

for lineNum, line in enumerate(lines):
    print("line: " + str(line))
    latexTableLine = ""

    key = line[header.index('key')]
    avgScore = safeConvert(line[header.index('avgScore')])
    avgCompleted = safeConvert(line[header.index('avgCompleted')])
    avgKnowledgeScore = safeConvert(line[header.index('avgKnowledgeManual')])
    avgTotalActions = safeConvert(line[header.index('avgTotalActions')])
    avgMoveActions = safeConvert(line[header.index('avgMoveActions')])
    avgNonMoveActions = safeConvert(line[header.index('avgNonMoveActions')])
    avgPropMoveActions = safeConvert(line[header.index('avgPropMoveActions')])
    avgavgPropNonMoveActions = safeConvert(line[header.index('avgPropNonMoveActions')])
    numSamples = safeConvert(line[header.index('numSamples')])

    knowledgeScore = 0

    # Get the topic and difficulty
    if (key not in LUT):
        print("ERROR: Key not found in LUT: " + key)
        continue

    lutRecord = LUT[key]
    topic = lutRecord["topic"]
    difficulty = lutRecord["difficulty"]
    taskDesc = lutRecord["taskDesc"]
    taskID = lutRecord["taskID"]

    #latex_table += r"\multicolumn{2}{l}{\textbf{" + topic + "}} & \multicolumn{11}{l}{" + taskDesc + "} &  \\\\ \n"

    color_row = "FFFFFF"
    bgcolRows = [2, 3, 8, 9, 14, 15, 20, 21]
    if (taskID in bgcolRows):
        color_row = "E3E3E3"

    latexTableLine += f"\\rowcolor[HTML]{{{color_row}}} \n"

    #color_row = 'D3D3D3' if row_num % 2 == 0 else 'FFFFFF'
    #latex_table += r"\rowcolor[HTML]{" + color_row + r"}\n"
    #latex_table += f"{row_num}  & {difficulty} & {task} "

    # Add key
    #fields = key.split("_")
    #topic = fields[0]
    #difficulty = fields[1]
    latexTableLine += f"{taskID} & {topic} & {difficulty} "

    # Add scores (performance, completion, knowledge)
    latexTableLine += f"& \\cellcolor[HTML]{{{get_scaled_color(avgScore, colors[0])}}}{dec2(avgScore)} "
    latexTableLine += f"& \\cellcolor[HTML]{{{get_scaled_color(avgCompleted, colors[1])}}}{dec2(avgCompleted)} "
    latexTableLine += f"& \\cellcolor[HTML]{{{get_scaled_color(avgKnowledgeScore, colors[2])}}}{dec2(avgKnowledgeScore)} & "

    # Add Action scores
    latexTableLine += f"& {int(avgTotalActions)} "
    latexTableLine += f"& {int(avgMoveActions)} "
    latexTableLine += f"& {int(avgNonMoveActions)} "
    # Add proportions
    #latex_table += f"& {dec2(avgPropMoveActions)} "
    latexTableLine += f"& {dec2(avgavgPropNonMoveActions)} & "

    # Add number of samples
    latexTableLine += f"& {int(numSamples)} "

#    if (lineNum % 4 == 3):
#        latex_table += r"\hline"

    # If the difficulty is "challenge", add an hline
    if (difficulty == "Challenge"):
        latexTableLine += "\\\\ \hline\n"
    else:
        latexTableLine += "\\\\ \n"

    # Add the line to the table
    linesToExport.append({
        "taskID": taskID,
        "line": latexTableLine
    })

    print("Processed")
    lineNum += 1


# Sort lines by task ID
linesToExport = sorted(linesToExport, key=lambda x: x["taskID"])
# Export the lines
for line in linesToExport:
    latex_table += line["line"]

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
filenameOut = "table-human.tex"
with open(filenameOut, 'w') as file:
    file.write(latex_table)

print("LaTeX table generated and saved as " + filenameOut)
