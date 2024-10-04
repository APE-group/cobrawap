"""
This is a parser for block scripts,
to produce the corresponding CWL CommandLineTool file
"""

import argparse
import os
from pathlib import Path
from cwl_utils import (
    parse_CLI_args,
    write_block_clt
)

CLI = argparse.ArgumentParser()
CLI.add_argument(
    "--block",
    nargs="?",
    type=Path,
    required=True,
    help="path to script file for which produce the CWL CommandLineTool file"
)

if __name__ == '__main__':

    args, unknown = CLI.parse_known_args()

    block_path = args.block.resolve() # .expanduser().absolute()
    block_name = block_path.stem
    block_yaml_path = Path(block_path.parent / f"{block_name}.yaml").resolve()

    cwl_path = Path(block_path.parents[1] / "cwl_steps").resolve()

    if not os.path.isdir(cwl_path):
        os.mkdir(cwl_path)
    cwl_file = Path(cwl_path / f"{block_name}.cwl").resolve()

    block_args = parse_CLI_args(block_path)

    write_block_clt(cwl_file, block_args)
