#!/usr/bin/env python
# encoding: utf8
'''
Collaborative Brain Wave Analysis Pipeline (Cobrawap)
'''

import os
import sys
import logging
import argparse
import subprocess
import shutil
import re
from pprint import pformat
from pathlib import Path
import inspect
sys.path.append(str(Path(inspect.getfile(lambda: None)).parent))
sys.path.append(str(Path(inspect.getfile(lambda: None)).parent / 'pipeline'))
from cmd_utils import get_setting, set_setting, get_initial_available_stages
from cmd_utils import is_profile_name_valid, create_new_configfile
from cmd_utils import input_profile, get_profile, setup_entry_stage
from cmd_utils import working_directory, load_config_file, get_config
from cmd_utils import locate_str_in_list, read_stage_output
from utils.cwl_clt_parser import write_block_yaml
log = logging.getLogger()
logging.basicConfig(level=logging.INFO)


try:
    STAGES = get_setting('stages')
except Exception as e:
    log.debug(e)
    try:
        STAGES = get_initial_available_stages()
    except Exception as e:
        log.debug(e)
        STAGES = {}

CLI = argparse.ArgumentParser(prog='cobrawap')

def get_parser():
    return CLI

# Utility arguments
CLI.add_argument("-v", "--verbose", action='store_true',
                help="print additional logging information")
CLI.add_argument("-V", "--version", action='version')
CLI.set_defaults(command=None)

# Initialization
subparsers = CLI.add_subparsers(help='')
CLI_init = subparsers.add_parser('init',
                help='initialize the cobrawap directories (required only once)')
CLI_init.set_defaults(command='init')
CLI_init.add_argument("--output_path", type=Path, default=None,
                      help="directory where the analysis output is stored "
                           "[default: '~/cobrawap_output/']")
CLI_init.add_argument("--config_path", type=Path, default=None,
                      help="directory where the analysis config files are "
                           "stored [default: '~/cobrawap_config/']")

# Show Settings
CLI_settings = subparsers.add_parser('settings',
                        help='display the content of ~/.cobrawap/config')
CLI_settings.set_defaults(command='settings')


# Configuration
CLI_create = subparsers.add_parser('create',
                        help='create configuration for a new dataset')
CLI_create.set_defaults(command='create')
CLI_create.add_argument("--data_path", type=Path, nargs='?', default=None,
                        help="full path to the dataset to be analyzed "
                             "(or where it will be stored)")
CLI_create.add_argument("--loading_script_name", nargs='?', type=Path, default=None,
                        help="name of the data specific loading script "
                             "(in <config_path>/stage01_data_entry/scripts/)")
CLI_create.add_argument("--profile", type=str, nargs='?', default=None,
                        help="profile name of this dataset/application "
                             "(see profile name conventions in documentation)")
CLI_create.add_argument("--parent_profile", type=str, nargs='?', default=None,
                        help="optional parent profile name "
                             "(see profile name conventions in documentation)")

# Additional configurations
CLI_profile = subparsers.add_parser('add_profile',
                         help='create a new configuration for an existing dataset')
CLI_profile.set_defaults(command='add_profile')
CLI_profile.add_argument("--profile", type=str, nargs='?', default=None,
                        help="profile name of this dataset/application "
                             "(see profile name conventions in documentation)")
CLI_profile.add_argument("--stages", type=str, nargs='*',
                         choices=list(STAGES.keys()), default=None,
                         help="selection of pipeline stages to configure")
CLI_profile.add_argument("--parent_profile", type=str, nargs='?', default=None,
                         help="optional parent profile name from which to "
                              "initialize the new config "
                              "[default: basic template]")

# Run
CLI_run = subparsers.add_parser('run',
                            help='run the analysis pipeline on the selected '
                                 'input and with the specified configurations')
CLI_run.set_defaults(command='run')
CLI_run.add_argument("--profile", type=str, nargs='?', default=None,
                     help="name of the config profile to be analyzed")
CLI_run.add_argument("--workflow_manager", type=str, nargs='?',
                     choices=['snakemake','cwl'], default='snakemake',
                     help="name of the workflow manager to use")

# Stage
CLI_stage = subparsers.add_parser('run_stage',
                                  help='execute an individual stage')
CLI_stage.set_defaults(command='run_stage')
CLI_stage.add_argument("--profile", type=str, nargs='?', default=None,
                       help="name of the config profile to be analyzed")
CLI_stage.add_argument("--stage", type=str, nargs='?', default=None,
                       choices=list(STAGES.keys()),
                       help="select individual stage to execute")
CLI_stage.add_argument("--workflow_manager", type=str, nargs='?',
                       choices=['snakemake','cwl'], default='snakemake',
                       help="name of the workflow manager to use")

# Block
CLI_block = subparsers.add_parser('run_block',
                        help='execute an individual block method on some input')
CLI_block.set_defaults(command='run_block')
CLI_block.add_argument("--block", type=str, nargs='?', default=None,
                       help="block specified as <stage_name>.<block_name>")
CLI_block.add_argument("--block_help", action='store_true',
                       help="display the help text of the block script")
CLI_block.add_argument("--workflow_manager", type=str, nargs='?',
                       choices=['snakemake','cwl'], default='snakemake',
                       help="name of the workflow manager to use")

def main():
    'Start main CLI entry point.'
    args, unknown = CLI.parse_known_args()
    # log.info("this is a regular print statement")
    # log.debug("this is a verbose print statement")

    if args.verbose:
        log.setLevel(logging.DEBUG)
    log.debug(pformat(args))

    if args.command == 'init':
        log.info("initializing Cobrawap")
        initialize(**vars(args))

    elif args.command == 'settings':
        log.info("display settings at ~/.cobrawap/config")
        print_settings(**vars(args))

    elif args.command == 'create':
        log.info("creating a set of config files")
        create(**vars(args))

    elif args.command == 'add_profile':
        log.info("creating a new config files")
        add_profile(**vars(args))

    elif args.command == 'run':
        log.info("executing Cobrawap")
        run(**vars(args), extra_args=unknown)

    elif args.command == 'run_stage':
        log.info("executing Cobrawap stage")
        run_stage(**vars(args), extra_args=unknown)

    elif args.command == 'run_block':
        log.info("executing Cobrawap block")
        run_block(**vars(args), block_args=unknown)

    elif args.command is None:
        CLI.print_help(sys.stderr)
        CLI.parse_args()

    else:
        log.info(f"{args.command} not known!")

    return None


def initialize(output_path=None, config_path=None, **kwargs):
    # set output_path
    if output_path is None:
        output_path = Path(input("Output directory "\
                                 "[default: ~/cobrawap_output]:")
                        or Path('~') / 'cobrawap_output').expanduser().resolve()
    output_path.mkdir(exist_ok=True)
    if not output_path.is_dir():
        raise ValueError(f"{output_path} is not a valid directory!")

    set_setting(dict(output_path=str(output_path)))

    # set config_path
    if config_path is None:
        config_path = Path(input("Config directory "\
                                 "[default: ~/cobrawap_configs]: ") \
                      or Path('~') / 'cobrawap_configs').expanduser().resolve()
    config_path.mkdir(parents=True, exist_ok=True)
    if not config_path.is_dir():
        raise ValueError(f"{config_path} is not a valid directory!")

    set_setting(dict(config_path=str(config_path)))

    # set pipeline path
    pipeline_path = Path(__file__).parents[1] / 'cobrawap' / 'pipeline'
    set_setting(dict(pipeline_path=str(pipeline_path.resolve())))

    # set available stages
    set_setting(dict(stages=get_initial_available_stages()))
    stages = get_setting('stages')

    # populate config_path with template config files
    if any(config_path.iterdir()):
        overwrite = (input(f"The config directory {config_path} already exists "\
                            "and is not empty. Create template config files "\
                            "anyway? [y/N]").lower() == 'y'
                     or False)
    else:
        overwrite = True
    if overwrite:
        for stage_number, stage in stages.items():
            stage_config_path = config_path / stage / 'configs'
            stage_config_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(pipeline_path / stage / 'configs' \
                                              / 'config_template.yaml',
                        stage_config_path / 'config.yaml')

        pipeline_config_path = config_path / 'configs'
        pipeline_config_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(pipeline_path / 'configs' / 'config_template.yaml',
                    pipeline_config_path / 'config.yaml')

        stage01_script_path = config_path / stages['1'] / 'scripts'
        stage01_script_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(pipeline_path / stages['1'] / 'scripts' \
                                                / 'enter_data_template.py',
                    stage01_script_path / 'enter_data_template.py')

    return None


def print_settings(*args, **kwargs):
    print(pformat(get_setting()))
    return None


def create(profile=None, parent_profile=None, data_path=None,
           loading_script_name=None, **kwargs):
    profile, parent_profile = get_profile(profile=profile,
                                          parent_profile=parent_profile)
    base_name = parent_profile if parent_profile else profile

    for stage_number, stage in get_setting('stages').items():
        config_name = profile if '1' in str(stage_number) else base_name
        create_new_configfile(stage=stage,
                              profile=config_name,
                              parent=parent_profile)

    setup_entry_stage(profile=profile,
                      parent_profile=parent_profile,
                      data_path=data_path,
                      loading_script_name=loading_script_name)
    return None


def add_profile(profile=None, parent_profile=None, stages=None,
                data_path=None, loading_script_name=None, **kwargs):
    profile, parent_profile = get_profile(profile=profile,
                                          parent_profile=parent_profile)
    # get stage selection
    stages = ''
    while not stages:
        stages = input("To which stages should this profile be applied? "
                      f"{list(get_setting('stages').keys())}:")
        try:
            stages = stages.replace("'","")
            stages = re.split(',|\s', stages)
            stages = [stage for stage in stages if stage]
        except Exception as e:
            log.info(e)
            stages = ''

    for stage_number in stages:
        create_new_configfile(stage_number=stage_number,
                              profile=profile,
                              parent=parent_profile)

    if any('1' in stage for stage in stages):
        setup_entry_stage(profile=profile, parent_profile=parent_profile,
                          data_path=data_path,
                          loading_script_name=loading_script_name)
    return None


def run(profile=None, workflow_manager='snakemake', extra_args=None, **kwargs):
    # select profile
    profile = input_profile(profile=profile)

    # set runtime config
    pipeline_path = Path(get_setting('pipeline_path'))

    if workflow_manager=='snakemake':

        # execute snakemake
        snakemake_args = ['snakemake', '-c1', '--config', f'PROFILE={profile}']
        log.info(f'Executing `{" ".join(snakemake_args+extra_args)}`')

        with working_directory(pipeline_path):
            subprocess.run(snakemake_args + extra_args)

    elif workflow_manager=='cwl':

        # execute cwl
        print('executing cwl')

    return None


def run_stage(profile=None, stage=None, workflow_manager='snakemake',
              extra_args=None, **kwargs):
    # select profile
    profile = input_profile(profile=profile)

    # get settings
    pipeline_path = Path(get_setting('pipeline_path'))
    config_path = Path(get_setting('config_path'))
    output_path = Path(get_setting('output_path'))
    stages = get_setting('stages')

    # select stage
    while stage not in stages.keys():
        stage = input("Which stage should be executed?\n    "
            +"\n    ".join(f"{k} {v}" for k,v in get_setting('stages').items())
            +"\nSelect the stage index: ")
    stage = stages[stage]

    # lookup stage input file
    pipeline_config_path = config_path / 'configs' / 'config.yaml'
    config_dict = load_config_file(pipeline_config_path)
    stage_idx = locate_str_in_list(config_dict['STAGES'], stage)
    # stage_idx_global = locate_str_in_list([v for k,v in stages.items()], stage)
    if stage_idx is None:
        raise IndexError("Make sure that the selected stage is also specified "\
                         "in your top-level config in the list `STAGES`!")

    stage_config_path = get_config(config_dir=config_path / stage,
                                   config_name=f'config_{profile}.yaml',
                                   get_path_instead=True)

    if stage_idx>0:
        prev_stage = config_dict['STAGES'][stage_idx-1]
        prev_stage_config_path = get_config(config_dir=config_path / prev_stage,
                                            config_name=f'config_{profile}.yaml',
                                            get_path_instead=True)
        prev_config_name = Path(prev_stage_config_path).name
        prev_output_name = read_stage_output(stage=prev_stage,
                                        config_dir=config_path,
                                        config_name=prev_config_name)
        stage_input = output_path / profile / prev_stage / prev_output_name
        extra_args = [f'STAGE_INPUT={stage_input}'] + extra_args

    # descend into stage folder
    stage_path = pipeline_path / stage

    # append stage specific arguments
    extra_args = extra_args + ['--configfile', f'{stage_config_path}']

    if workflow_manager=='snakemake':

        # execute snakemake
        snakemake_cl = ['snakemake', '-c1', '--config', f'PROFILE={profile}']
        snakemake_cl += extra_args
        log.info(f'Executing `{" ".join(snakemake_cl)}`')

        with working_directory(stage_path):
            subprocess.run(snakemake_cl)

    elif workflow_manager=='cwl':

        # write the cwl workflow file
        cwl_cl = ['python3', 'utils/cwl_wf_parser.py', '--stage', stage, \
                    '--configfile', f'{stage_config_path}']
        if stage_idx>0:
            cwl_cl += ['--stage_input', stage_input]
        log.info(f'Executing `{" ".join(cwl_cl)}`')
        myenv = os.environ.copy()
        myenv['PYTHONPATH'] = ':'.join(sys.path)
        with working_directory(pipeline_path):
            subprocess.run(cwl_cl, env=myenv)

        # execute the cwl workflow files
        # TBD...

    return None


def run_block(block=None, workflow_manager='snakemake', block_args=None,
              block_help=False, **kwargs):

    while not block:
        block = input("Specify a block to execute (<stage_name>.<block_name>):")

    stage, block  = re.split("\.|/|\s", block)[:2]
    stages = get_setting('stages')
    pipeline_path = Path(get_setting('pipeline_path'))

    while not stage in stages.values():
        print(f"Stage {stage} not found!\n"\
              f"Available stages are: {', '.join(list(stages.values()))}.")
        stage = input("Specify stage:")

    block_dir = pipeline_path / stage / 'scripts'
    available_blocks = [str(script.stem) for script in block_dir.iterdir() \
                        if (os.path.isfile(script) and script.suffix=='.py' and \
                            not str(script.stem).startswith('_') and \
                            'template' not in script.stem
                           )
                       ]

    while not block in available_blocks:
        print(f"Block {block} is not found in {stage}!\n"\
              f"Available blocks are: {', '.join(available_blocks)}.")
        block = input("Specify block:")

    if block_help:
        block_args += ['--help']

    myenv = os.environ.copy()
    myenv['PYTHONPATH'] = ':'.join(sys.path)
    if workflow_manager=='snakemake':
        # execute block
        snakemake_cl = ['python', str(block_dir / f'{block}.py')]
        snakemake_cl += block_args
        log.info(f'Executing `{" ".join(snakemake_cl)}`')
        with working_directory(pipeline_path):
            subprocess.run(snakemake_cl, env=myenv)
    elif workflow_manager=='cwl':
        # build the cwl step
        cwl_cl = ['python3', 'utils/cwl_clt_parser.py', \
                  '--block', str(block_dir / f'{block}.py')]
        log.info(f'Executing `{" ".join(cwl_cl)}`')
        with working_directory(pipeline_path):
            subprocess.run(cwl_cl, env=myenv)
        # build the yaml file from block_args
        block_args += ['--pipeline_path', pipeline_path]
        cwl_step_dir = pipeline_path / stage / 'cwl_steps'
        write_block_yaml(str(cwl_step_dir / f'{block}.yaml'), block_args)
        # execute the block
        cwl_cl = ['cwltool', str(cwl_step_dir / f'{block}.cwl'), \
                  str(cwl_step_dir / f'{block}.yaml')]
        log.info(f'Executing `{" ".join(cwl_cl)}`')
        with working_directory(pipeline_path):
            subprocess.run(cwl_cl, env=myenv)

    return None


if __name__ == '__main__':
    main()
