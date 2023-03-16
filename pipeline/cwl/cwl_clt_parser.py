"""
This is a parser for block scripts, to produce the corresponding CWL CommandLineTool file
"""

import argparse
import os
import yaml
import glob

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
  
def parse_CLI_args(script):
  raw_arg_lines = []
  with open(script, 'r') as f:
    whole_line = ''
    N_active_parenthesis = 0
    for l,line in enumerate(f):
      
      line = line.strip() # removes leading and trailing spaces
      line = line.partition('#')[0] # removes comments, if any
                                    # (also applies to comments added on the right,
                                    # just like these ones!)
      
      if "CLI.add_argument" in line:
        whole_line = line.partition('CLI.add_argument')[2]
        N_active_parenthesis = line.count('(') - line.count(')')
        if N_active_parenthesis == 0:
          raw_arg_lines.append(whole_line)
          whole_line = ''
      else:
        if whole_line != '':
          N_active_parenthesis += line.count('(') - line.count(')')
          if N_active_parenthesis > 0:
            whole_line += line
          elif N_active_parenthesis == 0:
            whole_line += line
            raw_arg_lines.append(whole_line)
            whole_line = ''
            N_active_parenthesis = 0
      
      if "CLI.parse_args()" in line:
        break
  
  args = []
  
  for raw_arg_line in raw_arg_lines:
    
    raw_arg = raw_arg_line.strip()
    #raw_arg = raw_arg.partition("CLI.add_argument(")[2]
    while raw_arg[0]=='(':
      raw_arg = raw_arg[1:]
    while raw_arg[-1]==')':
      raw_arg = raw_arg[:-1]
    #print('\nraw_arg:', raw_arg)
    arg_req = False
    for _ in raw_arg.split(','):
      _ = _.strip()
      if (_.startswith('"') and _.endswith('"')) or (_.startswith('\'') and _.endswith('\'')):
        _ = _[1:-1]
      if _[0:2]=='--':
        arg_name = _[2:]
      elif _.split('=')[0]=='type':
        arg_type = _.split('=')[1]
      elif _.split('=')[0]=='help':
        arg_help = _.split('=')[1]
        if (arg_help.startswith('"') and arg_help.endswith('"')) or (arg_help.startswith('\'') and arg_help.endswith('\'')):
          arg_help = arg_help[1:-1]
      elif _.split('=')[0]=='required' and _.split('=')[1]=='True':
        arg_req = True
    
    # mapping Python types into CWL types
    # CWL allowed types are: string, int, long, float, double, null, array, record, File, Directory, Any
    match arg_type:
      case 'str':
        arg_type = 'string'
      case 'int':
        arg_type = 'int'
      case 'float':
        arg_type = 'float'
      case _:
        arg_type = 'Any'
    
    if arg_name == 'data':
      arg_type = 'File'
    
    args.append({'name': arg_name, 'type': arg_type, 'required': arg_req, 'help': arg_help})
    
  return args

if __name__ == '__main__':
  
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--block", nargs='?', type=str, required=True,
                     help="path to script file(s) for which produce the CWL CommandLineTool file")
    args = CLI.parse_args()
    #args, unknown = CLI.parse_known_args()
    
    block_lists = glob.glob(args.block)
    block_lists = [_ for _ in block_lists if os.path.isfile(_)]
    block_lists = [_ for _ in block_lists if _.split('.')[-1]=='py']
    block_lists = [_ for _ in block_lists if 'template' not in _]
    
    if len(block_lists)==0:
      raise Exception(' > There are no python scripts in the provided path.')
    
    for block_script in block_lists:
      
      block_name = block_script.split('/')[-1].split('.')[0]
      block_parentpath = '/'.join(block_script.split('/')[:-1]) + '/'
      block_yaml = block_parentpath + block_name + ".yaml"
      print('\nblock_name: ' + block_name)
      
      cwl_path = block_parentpath + '../cwl_steps/'
      if not os.path.isdir(cwl_path):
        os.mkdir(cwl_path)
      cwl_file = cwl_path + block_name + '.cwl'
      
      args_1 = parse_CLI_args(block_script)
      #print('from parsing', args_1)
      
      if not os.path.isfile(block_yaml):
        print(' > There is no \'.yaml\' file for block script \'' + block_name + '\'.')
      else:
        args_2 = read_args_from_yaml(block_yaml)
        #print('from yaml', args_2)
        #print('parse == yaml?', args_1 == args_2)
            
        with open(cwl_file, "w+") as f_out:
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
          for a,arg in enumerate(args_1):
            f_out.write('  ' + arg['name'] + ':\n')
            f_out.write('    type: ' + arg['type'])
            if not arg['required']:
              f_out.write('?')
            f_out.write('\n')
            f_out.write('    inputBinding:\n')
            f_out.write('      position: ' + str(a+1) + '\n')
            f_out.write('      prefix: --' + arg['name'] + '\n')
          f_out.write('\n')
          f_out.write('outputs:\n')
          f_out.write('  ' + block_name + '_out:\n')
          f_out.write('    type: File\n')
          f_out.write('    outputBinding:\n')
          f_out.write('      glob: $(inputs.output)\n')

print('')
