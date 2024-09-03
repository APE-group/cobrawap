#import argparse
from cmd_utils import (
    get_setting,
    get_available_blocks
)

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

def write_block_clt(file_path, args):
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

def stage_block_list(stage, yaml_config):

    block_dir = Path(get_setting('pipeline_path')) / stage / "scripts"
    available_blocks = get_available_blocks(block_dir)

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
        # shouldn't be 'File' for 'data' ?
        #args.append({'name': arg_name, 'type': arg_type, 'required': arg_req, 'help': arg_help})
    return args
