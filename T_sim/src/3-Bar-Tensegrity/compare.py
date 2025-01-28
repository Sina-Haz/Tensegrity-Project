import subprocess
import re
import numpy as np

def parse_output(output):
    """Parse the program output into a structured dictionary."""
    data = {}
    lines = output.splitlines()
    timestep = None
    for line in lines:
        if line.startswith("timestep:"):
            timestep = int(line.split(":")[1].strip())
            data[timestep] = {}
        elif line.startswith("Body"):
            body = int(line.split()[1].strip(":"))
            data[timestep][body] = {}
        elif ":" in line:
            key, value = line.split(":")
            key = key.strip()
            # Updated regex to handle scientific notation
            value = np.array([float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", value)])
            data[timestep][body][key] = value
    return data


def compute_differences(data1, data2):
    """Compute differences between the outputs of two programs."""
    differences = {}
    for timestep in data1:
        differences[timestep] = {}
        for body in data1[timestep]:
            differences[timestep][body] = {}
            for key in data1[timestep][body]:
                diff = data1[timestep][body][key] - data2[timestep][body][key]
                differences[timestep][body][key] = diff
    return differences

def format_differences(differences):
    """Format the differences for printing."""
    output = []
    for timestep, bodies in differences.items():
        output.append(f"timestep: {timestep}")
        for body, metrics in bodies.items():
            output.append(f"Body {body}:")
            for key, value in metrics.items():
                formatted_values = " ".join(f"{x:+.3f}" for x in value)
                output.append(f"  {key}: {formatted_values}")
    return "\n".join(output)

# Run both scripts and capture their outputs
output_ti = subprocess.check_output(["/usr/local/bin/python3", "ti_3bar.py"], text=True)
output_mjc = subprocess.check_output(["/usr/local/bin/python3", "mjc-2-3bar.py"], text=True)

# Parse the outputs
data_ti = parse_output(output_ti)
data_mjc = parse_output(output_mjc)
# print(data_ti)
# print();print()
# print(data_mjc)
# Compute the differences
differences = compute_differences(data_ti, data_mjc)

# Print the formatted differences
print(format_differences(differences))
