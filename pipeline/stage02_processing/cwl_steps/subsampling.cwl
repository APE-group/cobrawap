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
      location: "../scripts/subsampling.py"
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
  target_rate:
    type: float
    inputBinding:
      position: 3
      prefix: --target_rate

outputs:
  subsampling_output:
    type: File
    outputBinding:
      glob: $(inputs.output)
