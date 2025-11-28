from score import scorer
from pathlib import Path
import os

input_file = "/home/rainfern/ALL_PROJECTS/2025 - FEEG_scorer/NIDRA/test/compare/LDTLR32024n2_ALL.edf"

output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_results") #Path('out')#

model = "all" # 'usleep', 'deepresnet', 'transformer', or 'all'

scorer(input_file, output_folder, model)
