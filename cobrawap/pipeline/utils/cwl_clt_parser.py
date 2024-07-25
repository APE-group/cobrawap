"""
This is a parser for block scripts, to produce the corresponding CWL CommandLineTool file
"""

import argparse
import os
import yaml
import glob
import sys
from pathlib import Path
import importlib

CLI = argparse.ArgumentParser()
CLI.add_argument("--block", nargs='+', type=Path, required=True,
                 help="path to script file(s) for which produce the CWL CommandLineTool file")

def pythontype_to_cwltype(python_type):
    if python_type is str:
        cwl_type = 'string'
    elif python_type is int:
        cwl_type = 'int'
    elif python_type is float:
        cwl_type = 'float'
    else:
        cwl_type = 'Any'
    return cwl_type

def read_args_from_yaml(yaml_file):
  with open(yaml_file, "r") as f:
    try:
      yaml_config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
      print(exc)
  for arg in yaml_config['args']:
    if arg['required'] == 'True':
      arg['required'] = True
    elif arg['required'] == 'False':
      arg['required'] = False
  return yaml_config['args']

def parse_CLI_args(block_path):
    block_name = block_path.stem
    #block_CLI = __import__(str(Path(block_name).expanduser().stem)).CLI
    #block_CLI = importlib.import_module(str(Path(block_name).expanduser().stem)).CLI
    # importing CLI dynamically from block script
    spec = importlib.util.spec_from_file_location(block_name, block_path)
    block = importlib.util.module_from_spec(spec)
    sys.modules[block_name] = block
    spec.loader.exec_module(block)
    block_CLI = block.CLI
    '''
    args = []
    for arg in block_CLI._actions:
        if type(arg) is argparse._StoreAction:
            # this arg is good to be parsed
            print(arg)
            args.append(arg.__dict__)
            #args.append({'name': arg.dest,'type': arg.type,'required': arg.required,'help': arg.help})
    '''
    args = [arg.__dict__ for arg in block_CLI._actions if type(arg) is argparse._StoreAction]
    for arg in args:
        # mapping Python types into CWL types
        # CWL allowed types are: string, int, long, float, double, null, array, record, File, Directory, Any
        arg['type'] = pythontype_to_cwltype(arg['type']) if arg['dest']!='data' else 'Any'
        #args.append({'name': arg_name, 'type': arg_type, 'required': arg_req, 'help': arg_help})
    return args

def write_clt_file(file_path, args):
    block_name = file_path.stem
    with open(file_path, "w+") as f_out:
        f_out.write('#!/usr/bin/env cwltool\n')
        f_out.write('\n')
        f_out.write('cwlVersion: v1.2\n')
        f_out.write('class: CommandLineTool\n')
        f_out.write('\n')
        f_out.write('baseCommand: python3\n')
        f_out.write('\n')
        f_out.write('requirements:\n')
        f_out.write('  EnvVarRequirement:\n')
        f_out.write('    envDef:\n')
        f_out.write('      PYTHONPATH: $(inputs.pipeline_path)\n')
        f_out.write('\n')
        f_out.write('inputs:\n')
        f_out.write('  pipeline_path:\n')
        f_out.write('    type: string\n')
        f_out.write('  step:\n')
        f_out.write('    type: File?\n')
        f_out.write('    default:\n')
        f_out.write('      class: File\n')
        f_out.write('      location: \"../scripts/' + block_name + '.py\"\n')
        f_out.write('    inputBinding:\n')
        f_out.write('      position: 0\n')
        for a,arg in enumerate(args):
            if 'name' not in arg.keys():
                arg['name'] = arg['dest']
            f_out.write('  ' + arg['name'] + ':\n')
            f_out.write('    type: ' + arg['type'])
            if not arg['required']:
                f_out.write('?')
            f_out.write('\n')
            f_out.write('    inputBinding:\n')
            f_out.write('      position: ' + str(a+1) + '\n')
            f_out.write('      prefix: --' + arg['name'] + '\n')
        f_out.write('\n')
        if 'output' in [_['name'] for _ in args]:
            f_out.write('outputs:\n')
            f_out.write('  ' + block_name + '_output:\n')
            f_out.write('    type: File\n')
            f_out.write('    outputBinding:\n')
            f_out.write('      glob: $(inputs.output)\n')
        else:
            f_out.write('outputs: []\n')

def write_block_yaml(file_path, block_args):
    print('block_args BEFORE:', block_args)
    block_args = [[_] if '=' not in str(_) else str(_).split('=') for _ in block_args]
    block_args = [arg for _ in block_args for arg in _]
    print('block_args AFTER:', block_args)
    #block_name = file_path.stem
    arg_dict = {}
    key = None
    for i,arg in enumerate(block_args):
        if str(arg).startswith('--'):
            key = arg[2:]
            key_count = 0
        else:
            if key and key_count == 0:
                arg_dict[key] = arg
                key_count += 1
            elif key and key_count == 1:
                arg_dict[key] = [arg_dict[key], arg]
                key_count += 1
            elif key and key_count > 1:
                arg_dict[key].append(arg)
                key_count += 1
    with open(file_path, "w+") as f_out:
        for key in arg_dict.keys():
            if key=='data':
                data_path = Path(arg_dict['data']).expanduser().resolve()
                f_out.write('data:\n')
                f_out.write('  class: File\n')
                f_out.write('  location: %s\n' % data_path)
            elif isinstance(arg_dict[key],list):
                f_out.write('%s:\n' % key)
                for val in arg_dict[key]:
                    f_out.write('  - %s\n' % val)
            else:
                f_out.write('%s: %s\n' % (key, arg_dict[key]))
        f_out.write('\n')

if __name__ == '__main__':

    args, unknown = CLI.parse_known_args()

    print('args.block:')
    for i,_ in enumerate(args.block):
        print(i, _)
    #block_lists = [Path(_).expanduser().absolute() for _ in args.block]
    block_lists = [_.resolve() for _ in args.block]
    #block_lists = glob.glob(args.block)
    print('block_lists:')
    for i,_ in enumerate(block_lists):
        print(i, _)

    block_lists = [_ for _ in block_lists
                   if (os.path.isfile(_) and _.suffix=='.py' and \
                       not str(_.stem).startswith('_') and 'template' not in _.stem)
                  ]

    print('block_lists:')
    for i,_ in enumerate(block_lists):
        print(i, _)

    if len(block_lists)==0:
      raise Exception(' > There are no python scripts in the provided path.')

    for block_path in block_lists:

      block_name = block_path.stem
      block_yaml = Path(str(block_path.parent) + '/' + block_name + '.yaml').resolve()

      cwl_path = Path(str(block_path.parent) + '/../cwl_steps').resolve()
      if not os.path.isdir(cwl_path):
        os.mkdir(cwl_path)
      cwl_file = Path(str(cwl_path) + '/' + block_name + '.cwl').resolve()

      args = parse_CLI_args(block_path)
      print('\nfrom clever parsing:')
      for i,_ in enumerate(args):
          print(i, _)

      write_clt_file(cwl_file, args)

print('')
