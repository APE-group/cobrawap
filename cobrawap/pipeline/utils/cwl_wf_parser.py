"""
This is a parser for stage-specific config files,
to produce the corresponding CWL WorkFlow file
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
    write_cwl_stage_files
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

    """
    with open(args.configfile, "r") as f:
        try:
            stage_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise exc

    if stage_config["STAGE_NAME"] != stage:
        raise Exception("The loaded config file is not coherent with the stage specified.")

    myenv = os.environ.copy()
    myenv["PYTHONPATH"] = ":".join(sys.path)

    """

    try:
        # Here the list of blocks is built, ordered according to user directives
        """
        block_list = stage_block_list(stage, args.configfile)
        for b,block in enumerate(block_list):
            block_path = stage_path / "scripts" / f"{block['name']}.py"
            cwl_cl = ["python3", "utils/cwl_clt_parser.py", "--block", str(block_path)]
            with working_directory(pipeline_path):
                subprocess.run(cwl_cl, env=myenv)
        """
        write_cwl_stage_files(stage, args.configfile, args.stage_input)

    except Exception as exc:
        raise exc
