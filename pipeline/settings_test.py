import os

# path for generated data
output_path = os.path.join(os.path.expanduser('~'), '/Users/giuliadebonis/gdb/test-CoBraWAP/test-WH-FunctionalAreas/Output')
print(output_path)
output_path = os.path.join(os.path.expanduser('~'), '/gdb/test-CoBraWAP/test-WH-FunctionalAreas/Output')
print(output_path)

print(os.path.join(os.path.expanduser('~'),'/pippo'))
print(os.path.join(os.path.expanduser('~')))

# optional alternative path for config files
# directory must contain stageXY_<stage-name>/config_<PROFILE>.yaml
# if None uses the pipeline working directory
configs_dir = None
