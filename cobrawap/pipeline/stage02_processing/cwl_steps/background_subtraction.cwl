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
      location: "../scripts/background_subtraction.py"
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
  output_img:
    type: Any?
    inputBinding:
      position: 3
      prefix: --output_img
  output_array:
    type: Any?
    inputBinding:
      position: 4
      prefix: --output_array

outputs:
  background_subtraction_output:
    type: File
    outputBinding:
      glob: $(inputs.output)
