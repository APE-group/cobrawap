import argparse
import importlib
import os
import subprocess
import sys
import yaml
from cmd_utils import (
    get_setting,
    get_available_blocks,
    working_directory
)
from pathlib import Path

pipeline_path = Path(get_setting("pipeline_path"))

myenv = os.environ.copy()
myenv["PYTHONPATH"] = ":".join(sys.path)

# Block level

def pythontype_to_cwltype(arg):
    if arg["type"] is str:
        cwl_type = "string"
    elif arg["type"] is int:
        cwl_type = "int"
    elif arg["type"] is float:
        cwl_type = "float"
    elif arg["type"] is Path:
        cwl_type = "File"
    else:
        cwl_type = "Any"
    if arg["dest"] in ["data", "original_data"]:
        cwl_type = "File"
    #if arg["dest"] in ["img_dir"]:
    #    cwl_type = "Directory"
    if cwl_type!="Any" and arg["nargs"]=="+":
        cwl_type += "[]"
    if not arg["required"]:
        cwl_type += "?"
    return cwl_type

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
    """
    args = []
    for arg in block_CLI._actions:
        if type(arg) is argparse._StoreAction:
            # this arg is good to be parsed
            print(arg)
            args.append(arg.__dict__)
            #args.append({'name': arg.dest,'type': arg.type,'required': arg.required,'help': arg.help})
    """
    args = [arg.__dict__ for arg in block_CLI._actions if type(arg) is argparse._StoreAction]
    for arg in args:
        # mapping Python types into CWL types
        # CWL allowed types are: string, int, long, float, double, null, array, record, File, Directory, Any
        # Be careful with typing of "data" and "output"; are they Any, string, or File?
        # arg["type"] = pythontype_to_cwltype(arg["dest"], arg["type"], arg["nargs"]) if arg["dest"] not in ["data","output"] else "File"
        arg["type"] = pythontype_to_cwltype(arg)
    return args

def write_block_yaml(file_path, block_args):
    print(f"block_args BEFORE: {block_args}")
    block_args = [[_] if "=" not in str(_) else str(_).split("=") for _ in block_args]
    block_args = [arg for _ in block_args for arg in _]
    print(f"block_args AFTER: {block_args}")
    #block_name = file_path.stem
    arg_dict = {}
    key = None
    for i,arg in enumerate(block_args):
        # loop designed for CLI args
        if str(arg).startswith("--"):
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
            if key=="data":
                data_path = Path(arg_dict["data"]).expanduser().resolve()
                f_out.write("data:\n")
                f_out.write("  class: File\n")
                f_out.write(f"  location: {data_path}\n")
            elif isinstance(arg_dict[key],list):
                f_out.write(f"{key}:\n")
                for val in arg_dict[key]:
                    f_out.write(f"  - {val}\n")
            else:
                f_out.write(f"{key}: {arg_dict[key]}\n")
        f_out.write("\n")

def write_block_clt(file_path, args):
    block_name = file_path.stem
    with open(file_path, "w+") as f_out:
        f_out.write("#!/usr/bin/env cwltool\n")
        f_out.write("\n")
        f_out.write("cwlVersion: v1.2\n")
        f_out.write("class: CommandLineTool\n")
        f_out.write("\n")
        f_out.write("baseCommand: python3\n")
        f_out.write("\n")
        f_out.write("requirements:\n")
        f_out.write("  EnvVarRequirement:\n")
        f_out.write("    envDef:\n")
        f_out.write("      PYTHONPATH: $(inputs.pipeline_path)\n")
        f_out.write("\n")
        f_out.write("inputs:\n")
        f_out.write("  pipeline_path:\n")
        f_out.write("    type: string\n")
        f_out.write("  step:\n")
        f_out.write("    type: File?\n")
        f_out.write("    default:\n")
        f_out.write("      class: File\n")
        f_out.write(f"      location: \"../scripts/{block_name}.py\"\n")
        f_out.write("    inputBinding:\n")
        f_out.write("      position: 0\n")
        for a,arg in enumerate(args):
            if "name" not in arg.keys():
                arg["name"] = arg["dest"]
            f_out.write(f"  {arg['name']}:\n")
            f_out.write(f"    type: {arg['type']}\n")
            #if not arg["required"]:
            #    f_out.write("?")
            #f_out.write("\n")
            f_out.write("    inputBinding:\n")
            f_out.write(f"      position: {a+1}\n")
            f_out.write(f"      prefix: --{arg['name']}\n")
        f_out.write("\n")
        if "output" in [_["name"] for _ in args]:
            f_out.write("outputs:\n")
            f_out.write(f"  {block_name}.output:\n")
            f_out.write("    type: File\n")
            f_out.write("    outputBinding:\n")
            f_out.write("      glob: $(inputs.output)\n")
        else:
            f_out.write("outputs: []\n")

# Stage level

def stage_block_list(stage, stage_config_path):

    with open(stage_config_path, "r") as f:
        try:
            stage_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise exc

    block_dir = Path(get_setting("pipeline_path")) / stage / "scripts"
    available_blocks = get_available_blocks(block_dir)

    match stage:

        case "stage01_data_entry":
            curate_block = Path(stage_config["CURATION_SCRIPT"]).stem
            block_list = [{"name": curate_block,
                           "depends_on": "STAGE_INPUT"},
                          {"name": "check_input",
                           "depends_on": curate_block},
                          {"name": "plot_traces",
                          "depends_on": curate_block}]

        case "stage02_processing":
            depends_on = "STAGE_INPUT"
            block_list = [{"name": "check_input",
                           "depends_on": depends_on}]
            if "BLOCK_ORDER" in stage_config.keys():
                for b,block in enumerate(stage_config["BLOCK_ORDER"]):
                    block_list.append({"name": block,
                                       "depends_on": depends_on})
                    depends_on = block_list[-1]["name"]
            block_list.append({"name": "plot_power_spectrum",
                               "depends_on": depends_on})
            block_list.append({"name": "plot_processed_trace",
                               "depends_on": depends_on})

        case "stage03_trigger_detection":
            block_list = [{"name": "check_input",
                           "depends_on": "STAGE_INPUT"}]
            try:
                if stage_config["DETECTION_BLOCK"] in ["hilbert_phase", "minima"]:
                    detection_block = stage_config["DETECTION_BLOCK"]
                elif stage_config["DETECTION_BLOCK"]=="threshold":
                    if stage_config["THRESHOLD_METHOD"] in ["fixed", "fitted"]:
                        detection_block = f"calc_threshold_{stage_config['THRESHOLD_METHOD']}"
                filter_blocks = stage_config["TRIGGER_FILTER"]
                block_list.append({"name": detection_block,
                                   "depends_on": "STAGE_INPUT"})
                depends_on = detection_block
            except:
                raise KeyError("Check \"DETECTION\" fields in config file.")
            try:
                if "TRIGGER_FILTER" in stage_config.keys():
                    for b,block in enumerate(stage_config["TRIGGER_FILTER"]):
                        block_list.append({"name": block,
                                        "depends_on": depends_on})
                        depends_on = block_list[-1]["name"]
            except:
                raise KeyError("Check \"TRIGGER_FILTER\" field in config file.")
            block_list.append({"name": "plot_trigger_times",
                               "depends_on": depends_on})

        case "stage04_wave_detection":
            block_list = ["check_input"]
            # TBD

        case "stage05_channel_wave_characterization":
            block_list = ["check_input"]
            # TBD

        case "stage05_wave_characterization":
            block_list = ["check_input"]
            # TBD

    missing_blocks = [block["name"] for block in block_list if block["name"] not in available_blocks]
    if len(missing_blocks)>0:
        raise Exception(f"The following blocks are not available: {missing_blocks}")

    return block_list

wf_header = "#!/usr/bin/env cwltool\n\n" + \
            "cwlVersion: v1.2\n" + \
            "class: Workflow\n\n"

#def write_wf_file(stage_path, stage_input, block_list, yaml_config, global_input_list, detailed_input):
def write_wf_file(stage, stage_config_path, stage_input=None):

    stage_path = pipeline_path / stage

    block_list = stage_block_list(stage, stage_config_path)

    with open(stage_config_path, "r") as f:
        try:
            stage_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise exc

    if stage_config["STAGE_NAME"] != stage:
        raise Exception("The loaded config file is not coherent with the stage specified.")

    detailed_input = {}
    global_input_list = [
        {"block_name": "", "input_name": "data", "input_name_with_prefix": "data", "input_type": "Any"},
        {"block_name": "", "input_name": "pipeline_path", "input_name_with_prefix": "pipeline_path", "input_type": "string"}
    ]
    block_specs = {}
    for b,blk in enumerate(block_list):

        # write the clt file
        print(blk)
        block = blk["name"]
        block_path = stage_path / "scripts" / f"{block}.py"
        cwl_cl = ["python3", "utils/cwl_clt_parser.py", "--block", str(block_path)]
        with working_directory(pipeline_path):
            subprocess.run(cwl_cl, env=myenv)

        # use the clt file
        block_specs[block] = {}
        detailed_input[block] = []
        cwl_step = stage_path / "cwl_steps" / f"{block}.cwl"
        with open(cwl_step, "r") as f:
            try:
                yaml_block_file = yaml.safe_load(f)
                block_inputs = list(yaml_block_file["inputs"].keys())
                has_output = False if len(yaml_block_file["outputs"])==0 else True
            except yaml.YAMLError as exc:
                raise exc

            block_inputs = [_ for _ in block_inputs if _ not in ["pipeline_path","step"]]

            block_specs[block]["depends_on"] = [_["depends_on"] for _ in block_list if _["name"]==block][0]
            block_specs[block]["inputs"] = {}
            for input in block_inputs:
                block_specs[block]["inputs"][input] = yaml_block_file["inputs"][input]
                detailed_input[block].append(input)
                global_input_list.append({"block_name": block,
                                          "input_name": input,
                                          "input_name_with_prefix": f"{block}_{input}",
                                          "input_type": yaml_block_file["inputs"][input]["type"]
                                         })
            block_specs[block]["has_output"] = has_output

    print("\nglobal_list:\n", global_input_list, "\n")
    print("detailed_list:\n", detailed_input, "\n")
    print("block_specs:\n", block_specs, "\n")

    wf_path = stage_path / "workflow_NEW.cwl"
    with open(wf_path, "w+") as f_out:

        # headers
        f_out.write(wf_header)

        # inputs
        f_out.write("inputs:\n\n")
        f_out.write(f"  # General\n")
        f_out.write("  pipeline_path: string\n\n")
        for b,blk in enumerate(block_list):
            block = blk["name"]
            f_out.write(f"  # Block \'{block}\'\n")
            for input in block_specs[block]["inputs"].keys():
                if input!="data" or (input=="data" and block_specs[block]["depends_on"]=="STAGE_INPUT"):
                    f_out.write(f"  {block}.{input}: {block_specs[block]['inputs'][input]['type']}\n")
            f_out.write("\n")

        # outputs
        last_output = None
        f_out.write("outputs:\n\n")
        for b,blk in enumerate(block_list):
            block = blk["name"]
            if block_specs[block]["has_output"]:
                f_out.write(f"  {block}.output:\n")
                f_out.write(f"    type: File\n")
                f_out.write(f"    outputSource: {block}/{block}.output\n")
                f_out.write("\n")
                last_output = f"{block}/{block}.output"
        if last_output is not None:
            f_out.write("  final_output:\n")
            f_out.write("    type: File\n")
            f_out.write(f"    outputSource: {last_output}\n\n")

        # steps
        f_out.write("steps:\n\n")
        for b,blk in enumerate(block_list):
            block = blk["name"]
            f_out.write(f"  {block}:\n")
            f_out.write(f"    run: cwl_steps/{block}.cwl\n")
            f_out.write(f"    in:\n")
            f_out.write("      pipeline_path: pipeline_path\n")
            for input in block_specs[block]["inputs"].keys():
                if input=="data":
                    if blk["depends_on"]=="STAGE_INPUT":
                        f_out.write(f"      data: {block}.data\n")
                    else:
                        f_out.write(f"      data: {blk['depends_on']}/{blk['depends_on']}.output\n")
                else:
                    f_out.write(f"      {input}: {block}.{input}\n")
            if block_specs[block]["has_output"]:
                f_out.write(f"    out: [{block}.output]\n")
            else:
                f_out.write("    out: []\n")
            f_out.write("\n")


    wf_path = stage_path / "workflow.cwl"
    with open(wf_path, "w+") as f_out:

        # headers
        f_out.write(wf_header)

        # inputs
        f_out.write("inputs:\n\n")
        f_out.write("  pipeline_path: string\n\n")

        for inp in global_input_list:
            if inp["input_name"]=="output":
                f_out.write(f"  {inp['input_name_with_prefix']}:\n")
                f_out.write("    type: string?\n")
                if "plot" not in inp["block_name"]:
                    # NEO OUTPUT
                    f_out.write(f"    default: \"{inp['block_name']}.{stage_config['NEO_FORMAT']}\"\n")
                else:
                    # PLOT OUTPUT
                    f_out.write(f"    default: \"plot.{stage_config['PLOT_FORMAT']}\"\n")
            else:
                f_out.write(f"  {inp['input_name_with_prefix']}: {inp['input_type']}\n")
        f_out.write("\n")
        f_out.write("outputs:\n")
        f_out.write("\n")
        last_output = None
        for b,blk in enumerate(block_list):
            block = blk["name"]
            if "output" in detailed_input[block]:
                f_out.write(f"  {block}_output:\n")
                f_out.write("    type: File\n")
                f_out.write(f"    outputSource: {block}/{block}_output\n")
                f_out.write("\n")
                last_output = f"{block}/{block}_output"
        if last_output is not None:
            f_out.write("  final_output:\n")
            f_out.write("    type: File\n")
            f_out.write("    outputSource: {last_output}\n")
        f_out.write("\n")
        f_out.write("steps:\n")
        f_out.write("\n")
        last_output = "data"
        for b,blk in enumerate(block_list):
            block = blk["name"]
            f_out.write(f"  {block}:\n")
            f_out.write(f"    run: cwl_steps/{block}.cwl\n")
            f_out.write("    in:\n")
            f_out.write("      pipeline_path: pipeline_path\n")
            f_out.write(f"      data: {last_output}\n")
            for inp in detailed_input[block]:
                f_out.write(f"      {inp}: {block}_{inp}\n")
            f_out.write(f"    out: [{block}_output]\n")
            if "output" in detailed_input[block]:
                last_output = f"{block}/{block}_output"
            f_out.write("\n")

    yaml_path = stage_path / "workflow_2.yaml"
    with open(yaml_path, "w+") as f_out:
        f_out.write(f"STAGE_NAME: \"{stage_path.stem}\"\n\n")
        f_out.write(f"pipeline_path: \"{Path(get_setting('pipeline_path'))}\"\n\n")
        # TBD: check what happens with stage_input when stage_idx==0
        f_out.write("# stage input\n")
        f_out.write("data:\n")
        f_out.write("    class: File\n")
        f_out.write(f"    location: {stage_input}\n\n")
        for b,blk in enumerate(block_list):
            block = blk["name"]
            print(block)
            print(detailed_input[block])
            print(stage_config.keys(), "\n")
            f_out.write(f"# block \"{block}\"\n")
            for inp in detailed_input[block]:
                write = True
                if inp=="data":
                    if blk["depends_on"]=="STAGE_INPUT":
                        value = f"\n    class: File\n    location: \"{stage_input}\""
                    else:
                        write = False
                elif inp=="original_data":
                    value = f"\n    class: File\n    location: \"{block}.{stage_config['NEO_FORMAT']}\""
                elif inp=="t_start":
                    value = stage_config["PLOT_TSTART"]
                elif inp=="t_stop":
                    value = stage_config["PLOT_TSTOP"]
                elif inp=="channels" and "PLOT_CHANNELS" in stage_config.keys():
                    value = stage_config["PLOT_CHANNELS"]
                elif inp.upper() in stage_config.keys():
                    value = stage_config[inp.upper()]
                else:
                    value = None
                if write:
                    f_out.write(f"{block}.{inp}: {value}\n")
            f_out.write("\n")

    return
