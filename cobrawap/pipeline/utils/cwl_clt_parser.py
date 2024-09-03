"""
This is a parser for block scripts, to produce the corresponding CWL CommandLineTool file
"""

import argparse
import os
from pathlib import Path
from cwl_utils import (
    parse_CLI_args,
    write_block_clt
)

CLI = argparse.ArgumentParser()
CLI.add_argument("--block", nargs='+', type=Path, required=True,
                 help="path to script file(s) for which produce the CWL CommandLineTool file")

if __name__ == '__main__':

    args, unknown = CLI.parse_known_args()

    #block_lists = [Path(_).expanduser().absolute() for _ in args.block]
    block_lists = [_.resolve() for _ in args.block]

    if len(block_lists)==0:
      raise Exception(' > There are no python scripts in the provided path.')

    for block_path in block_lists:

      block_name = block_path.stem
      block_yaml = Path(block_path.parent / f'{block_name}.yaml').resolve()

      cwl_path = Path(block_path.parents[1] / 'cwl_steps').resolve()
      print(cwl_path)
      if not os.path.isdir(cwl_path):
        os.mkdir(cwl_path)
      cwl_file = Path(cwl_path / f'{block_name}.cwl').resolve()

      args = parse_CLI_args(block_path)

      write_block_clt(cwl_file, args)

print('')
