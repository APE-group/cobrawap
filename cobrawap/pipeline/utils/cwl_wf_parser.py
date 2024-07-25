"""
This is a parser for stage-specific config files, to produce the corresponding CWL WorkFlow file
"""

import argparse
import os
import sys
import yaml
import glob
from pathlib import Path
import subprocess
from cmd_utils import get_setting
from cmd_utils import working_directory

CLI = argparse.ArgumentParser()
CLI.add_argument("--stage", nargs='?', type=str, required=True,
                 help="name of the stage for which produce the CWL WorkFlow file")
CLI.add_argument("--stage_input", nargs='?', type=Path, required=False,
                 help="path to the input file for the stage")
CLI.add_argument("--configfile", nargs='?', type=Path, required=True,
                 help="path to the config file for customizing the stage to be executed")

def stage_block_list(stage, yaml_config):

    block_dir = Path(get_setting('pipeline_path')) / stage / 'scripts'
    available_blocks = [str(script.stem) for script in block_dir.iterdir() \
                        if (os.path.isfile(script) and script.suffix=='.py' and \
                            not str(script.stem).startswith('_') and \
                            'template' not in script.stem
                           )
                       ]

    match stage:

        case 'stage01_data_entry':
            curate_block = yaml_config['CURATION_SCRIPT'].split('.')[0]
            block_list = [curate_block,'check_input','plot_traces']

        case 'stage02_processing':
            block_list = ['check_input']
            try:
                block_list.extend(yaml_config['BLOCK_ORDER'])
            except:
                raise Exception('\'BLOCK_ORDER\' field missing in the config file.')
            block_list.extend(['plot_power_spectrum','plot_processed_trace'])

    missing_blocks = [block for block in block_list if block not in available_blocks]
    if len(missing_blocks)>0:
        raise Exception('The following blocks are not available:', missing_blocks)

    return block_list

def write_wf_file(stage_path, stage_input, block_list, yaml_config, global_input_list, detailed_input):

    wf_path = stage_path / 'workflow.cwl'
    with open(wf_path, "w+") as f_out:
        f_out.write('#!/usr/bin/env cwltool\n')
        f_out.write('\n')
        f_out.write('cwlVersion: v1.2\n')
        f_out.write('class: Workflow\n')
        f_out.write('\n')
        f_out.write('inputs:\n')
        f_out.write('\n')
        for inp in global_input_list:
            if inp['input_name']=='output':
                f_out.write('  %s:\n' % inp['input_name_with_prefix'])
                f_out.write('    type: string?\n')
                if 'plot' not in inp['block_name']:
                    # NEO OUTPUT
                    f_out.write('    default: \"%s.%s\"\n' % (inp['block_name'], yaml_config['NEO_FORMAT']))
                else:
                    # PLOT OUTPUT
                    f_out.write('    default: \"plot.%s\"\n' % yaml_config['PLOT_FORMAT'])
            else:
                f_out.write('  %s: %s\n' % (inp['input_name_with_prefix'],inp['input_type']))
        f_out.write('\n')
        f_out.write('outputs:\n')
        f_out.write('\n')
        last_output = None
        for b,block in enumerate(block_list):
            if 'output' in detailed_input[block]:
                f_out.write('  %s_output:\n' % block)
                f_out.write('    type: File\n')
                f_out.write('    outputSource: %s/%s_output\n' % (block,block))
                f_out.write('\n')
                last_output = '%s/%s_output' % (block,block)
        if last_output is not None:
            f_out.write('  final_output:\n')
            f_out.write('    type: File\n')
            f_out.write('    outputSource: %s\n' % last_output)
        f_out.write('\n')
        f_out.write('steps:\n')
        f_out.write('\n')
        last_output = 'data'
        for b,block in enumerate(block_list):
            f_out.write('  %s:\n' % block)
            f_out.write('    run: cwl_steps/%s.cwl\n' % block)
            f_out.write('    in:\n')
            f_out.write('      pipeline_path: pipeline_path\n')
            f_out.write('      data: %s\n' % last_output)
            for inp in detailed_input[block]:
                f_out.write('      %s: %s_%s\n' % (inp,block,inp))
            f_out.write('    out: [%s_output]\n' % block)
            if 'output' in detailed_input[block]:
                last_output = '%s/%s_output' % (block,block)
            f_out.write('\n')

    yaml_path = stage_path / 'workflow_2.yaml'
    with open(yaml_path, "w+") as f_out:
        f_out.write('STAGE_NAME: \"%s\"\n\n' % str(stage_path.stem))
        f_out.write('pipeline_path: \"%s\"\n\n' % str(Path(get_setting('pipeline_path'))))
        f_out.write('# stage input\n')
        f_out.write('data:\n')
        f_out.write('    class: File\n')
        f_out.write('    location: %s\n\n' % str(stage_input))
        for b,block in enumerate(block_list):
            print(block)
            print(detailed_input[block])
            print(yaml_config.keys(),'\n')
            f_out.write('# block \"%s\"\n' % block)
            for inp in detailed_input[block]:
                if inp.upper() in yaml_config.keys():
                    value = yaml_config[inp.upper()]
                elif inp=='output':
                    value = '\"' + block + '.' + yaml_config['NEO_FORMAT'] + '\"'
                elif inp=='t_start':
                    value = yaml_config['PLOT_TSTART']
                elif inp=='t_stop':
                    value = yaml_config['PLOT_TSTOP']
                elif inp=='channels' and 'PLOT_CHANNELS' in yaml_config.keys():
                    value = yaml_config['PLOT_CHANNELS']
                else:
                    value = None
                f_out.write('%s_%s: %s\n' % (block, inp, value))
            f_out.write('\n')

    return


if __name__ == '__main__':

    args, unknown = CLI.parse_known_args()

    # should now be useless, as the check is already done in __main__.py
    """
    if args.stage in ['1','stage01_data_entry','data_entry']:
        stage_fullname = 'stage01_data_entry'
    elif args.stage in ['2','stage02_processing','processing']:
        stage_fullname = 'stage02_processing'
    elif args.stage in ['3','stage03_trigger_detection','trigger_detection']:
        stage_fullname = 'stage03_trigger_detection'
    elif args.stage in ['4','stage04_wave_detection','wave_detection']:
        stage_fullname = 'stage04_wave_detection'
    elif args.stage in ['5','stage05_channel-wave_characterization','channel-wave_characterization','stage05_wave_characterization','wave_characterization']:
        if args.stage in ['stage05_channel-wave_characterization','channel-wave_characterization']:
            stage_fullname = 'stage05_channel-wave_characterization'
        elif args.stage in ['stage05_wave_characterization','wave_characterization']:
            stage_fullname = 'stage05_wave_characterization'
    else:
        print('unrecognized stage name: %s' % args.stage)
        quit()
    """

    stage = args.stage
    pipeline_path = Path(get_setting('pipeline_path'))
    stage_path = pipeline_path / stage

    print('stage:', stage)

    with open(args.configfile, "r") as f:
        try:
            yaml_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise exc

    if yaml_config['STAGE_NAME'] != stage:
        raise Exception('The loaded config file is not coherent with the stage specified.')

    try:
        block_list = stage_block_list(stage, yaml_config)
        print('List of blocks to be executed:', block_list)

        detailed_input = {}
        global_input_list = [
            {'block_name': '', 'input_name': 'data', 'input_name_with_prefix': 'data', 'input_type': 'Any'},
            {'block_name': '', 'input_name': 'pipeline_path', 'input_name_with_prefix': 'pipeline_path', 'input_type': 'string'}
        ]
        for b,block in enumerate(block_list):
            print(block)
            # write the clt file
            block_path = stage_path / 'scripts' / Path(block + '.py')
            cwl_args = ['python3', 'utils/cwl_clt_parser.py', '--block', str(block_path)]
            myenv = os.environ.copy()
            myenv['PYTHONPATH'] = ':'.join(sys.path)
            with working_directory(pipeline_path):
                subprocess.run(cwl_args, env=myenv)

            # use the clt file
            detailed_input[block] = []
            cwl_step = stage_path / 'cwl_steps' / Path(block + '.cwl')
            with open(cwl_step, "r") as f:
                try:
                    y = yaml.safe_load(f)
                    z = list(y['inputs'].keys())
                    print(block, ':', z)
                except yaml.YAMLError as exc:
                    raise exc

                z = [_ for _ in z if _ not in ['pipeline_path','step','data']]

                for inp in z:
                    detailed_input[block].append(inp)
                    global_input_list.append({'block_name': block, 'input_name': inp, 'input_name_with_prefix': block+'_'+inp, 'input_type': y['inputs'][inp]['type']})

        print('\nglobal_list:\n', global_input_list, '\n')
        print('detailed_list:\n', detailed_input, '\n')

        write_wf_file(stage_path, args.stage_input, block_list, yaml_config, global_input_list, detailed_input)

    except Exception as exc:
        raise exc
