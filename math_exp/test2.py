import os
import re

def generate_latex_figures(folder_path):
    # List all files in the folder
    file_names = os.listdir(folder_path)

    # Filter files to match the specified pattern and extract checkpoint numbers
    filtered_files = [name for name in file_names if re.match(r'^\d+_What_is_3_plus_12_Answermlp', name)]
    checkpoint_numbers = sorted([re.findall(r'^(\d+)_', name)[0] for name in filtered_files], key=int)


    # Initialize LaTeX code
    latex_code = ""
    subfig_template = "\\includegraphics[width=0.2\\textwidth]{{figures/gpt2xl/allcheckpoint/{}_What_is_3_plus_12_Answermlp.png}}"

    # Iterate over checkpoints to create figures
    for i in range(0, len(checkpoint_numbers), 5):
        latex_code += "\\begin{figure}[!ht]\n    \\centering\n"
        subfigures = [subfig_template.format(checkpoint_numbers[j]) for j in range(i, min(i+5, len(checkpoint_numbers)))]
        latex_code += "\n".join(subfigures).rstrip('~')
        caption_range = f"from checkpoints {checkpoint_numbers[i]} to {checkpoint_numbers[min(i+4, len(checkpoint_numbers)-1)]}"
        latex_code +=f"\n    \\caption{{Sequence of images" + caption_range+ f".}}\n    \\label{{fig:sequence{str(i//5 + 1)}}}\n\\"+"end{figure}\n\n"
    return latex_code

# Folder path
folder_path = '/project/vsharan_1180/Tianyi/rome/my_exp_res3'

# Generate LaTeX code
latex_code = generate_latex_figures(folder_path)
print(latex_code)
