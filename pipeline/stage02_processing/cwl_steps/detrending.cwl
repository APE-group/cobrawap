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
      location: "../scripts/detrending.py"
    inputBinding:
      position: 0
  data:
    type: File
    inputBinding:
      position: 1
      prefix: --data
  output:
    type: string
    inputBinding:
      position: 2
      prefix: --output
  order:
    type: int?
    inputBinding:
      position: 3
      prefix: --order
  img_dir:
    type: string
    inputBinding:
      position: 4
      prefix: --img_dir
  img_name:
    type: string?
    inputBinding:
      position: 5
      prefix: --img_name
  channels:
    type: int?
    inputBinding:
      position: 6
      prefix: --channels

outputs:
  detrending_output:
    type: File
    outputBinding:
      glob: $(inputs.output)
