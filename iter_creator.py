import inspect
import re
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def get_iteration_space( func, *args):
        # Initialize empty lists to store the iteration variable values
    variable_values = [[] for _ in range(len(args))]

    # Get the function name as a string
    func_name = func.__name__

    # Get the source code of the function
    source_code = inspect.getsource(func)

    # Append the line to capture the iteration space after each for loop
    indentation_level = 0
    modified_lines = ["variable_values = {0: [] , 1 : [] , 2 : [] , 3 : [] , 4 : []}\niter_timestep = 0"]
    number_of_line = 0
    for line in source_code.splitlines():
        number_of_line = number_of_line + 1
        if number_of_line == 2:
            modified_lines.append("    global variable_values \n    global iter_timestep")
        modified_lines.append(line)
        if line.strip().startswith("for"):
            index_name = re.findall(r'for\s+(\w+)\s+in', line)[0]
            indentation_level = len(line) - len(line.strip())
            modified_lines.append(' ' * (indentation_level + 4) + f'variable_values[{indentation_level//4}].append(( iter_timestep := iter_timestep + 1  , {index_name} , {indentation_level//4}))')


    modified_code = '\n'.join(modified_lines)
    modified_code += f"\n{func_name}{args}"
    #modified_code += ("\nprint(variable_values)")


    # Execute the modified code in the local scope
    #print(modified_code)
    globals_dict = {}
    locals_dict = {}
    exec(modified_code, globals_dict, locals_dict)
    iter_space_dic = globals_dict["variable_values"]
    return iter_space_dic