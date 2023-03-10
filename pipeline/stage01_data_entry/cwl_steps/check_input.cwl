#!/usr/bin/env cwltool

cwlVersion: v1.2
class: CommandLineTool

baseCommand: python3

requirements:
  EnvVarRequirement:
    envDef:
      PYTHONPATH: $(inputs.pipeline_path)

inputs:
  pipeline_path:
    type: string
  step:
    type: File?
    default:
      class: File
      location: "../scripts/check_input.py"
    inputBinding:
      position: 0
  data:
    type: str
    inputBinding:
      position: 1
      prefix: --data
  img:
    type: str
    inputBinding:
      position: 2
      prefix: --img

outputs:
  check_input_out:
    type: File
    outputBinding:
      glob: $(inputs.check_input_out)
