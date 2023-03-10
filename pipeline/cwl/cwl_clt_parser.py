"""
This is a parser for block scripts, to produce the corresponding CWL CommandLineTool file
"""

import argparse
import os
import yaml
import glob

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
      
      if not os.path.isfile(block_yaml):
        print(' > There is no \'.yaml\' file for block script \'' + block_name + '\'.')
      else:
        with open(block_yaml, "r") as f:
          try:
            yaml_config = yaml.safe_load(f)
            
            cwl_path = block_parentpath + '../cwl_steps/'
            if not os.path.isdir(cwl_path):
              os.mkdir(cwl_path)
            cwl_file = cwl_path + block_name + '.cwl'
            
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
              for a,arg in enumerate(yaml_config['args']):
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
              f_out.write('      glob: $(inputs.' + block_name + '_out)\n')
          
          except yaml.YAMLError as exc:
            print(exc)

print('')
