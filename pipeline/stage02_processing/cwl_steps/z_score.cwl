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
      location: "../scripts/z_score.py"
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

outputs:
  z_score_output:
    type: File
    outputBinding:
      glob: $(inputs.output)
