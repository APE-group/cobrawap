"""
This is a parser for stage-specific config files, to produce the corresponding CWL WorkFlow file
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
import subprocess
from cmd_utils import (
    get_setting,
    working_directory
)
from cwl_utils import (
    stage_block_list,
    write_wf_file
)

CLI = argparse.ArgumentParser()
CLI.add_argument("--stage", nargs='?', type=str, required=True,
                 help="name of the stage for which produce the CWL WorkFlow file")
CLI.add_argument("--stage_input", nargs='?', type=Path, required=False,
                 help="path to the input file for the stage")
CLI.add_argument("--configfile", nargs='?', type=Path, required=True,
                 help="path to the config file for customizing the stage to be executed")

if __name__ == '__main__':

    args, unknown = CLI.parse_known_args()

    stage = args.stage
    pipeline_path = Path(get_setting("pipeline_path"))
    stage_path = pipeline_path / stage

    print("stage:", stage)

    with open(args.configfile, "r") as f:
        try:
            yaml_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise exc

    if yaml_config["STAGE_NAME"] != stage:
        raise Exception("The loaded config file is not coherent with the stage specified.")

    try:
        # Here the list of blocks is built, ordered according to user directives
        block_list = stage_block_list(stage, yaml_config)
        print("List of blocks to be executed:", block_list)

        detailed_input = {}
        global_input_list = [
            {'block_name': '', 'input_name': 'data', 'input_name_with_prefix': 'data', 'input_type': 'Any'},
            {'block_name': '', 'input_name': 'pipeline_path', 'input_name_with_prefix': 'pipeline_path', 'input_type': 'string'}
        ]
        for b,block in enumerate(block_list):
            print(block)
            # write the clt file
            block_path = stage_path / "scripts" / f"{block}.py"
            cwl_args = ["python3", "utils/cwl_clt_parser.py", "--block", str(block_path)]
            myenv = os.environ.copy()
            myenv["PYTHONPATH"] = ":".join(sys.path)
            with working_directory(pipeline_path):
                subprocess.run(cwl_args, env=myenv)

            # use the clt file
            detailed_input[block] = []
            cwl_step = stage_path / "cwl_steps" / f"{block}.cwl"
            with open(cwl_step, "r") as f:
                try:
                    y = yaml.safe_load(f)
                    z = list(y["inputs"].keys())
                    print(block, ":", z)
                except yaml.YAMLError as exc:
                    raise exc

                z = [_ for _ in z if _ not in ["pipeline_path","step","data"]]

                for inp in z:
                    detailed_input[block].append(inp)
                    global_input_list.append({'block_name': block, 'input_name': inp, 'input_name_with_prefix': block+'_'+inp, 'input_type': y['inputs'][inp]['type']})

        print('\nglobal_list:\n', global_input_list, '\n')
        print('detailed_list:\n', detailed_input, '\n')

        write_wf_file(stage_path, args.stage_input, block_list, yaml_config, global_input_list, detailed_input)

    except Exception as exc:
        raise exc
