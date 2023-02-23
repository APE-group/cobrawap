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
  input_file:
    type: File
    inputBinding:
      position: 1
      prefix: --data
  background_subtraction_out:
    type: string
    inputBinding:
      position: 2
      prefix: --output
  output_img:
    type: string?
    inputBinding:
      position: 3
      prefix: --output_img


outputs:
  background_subtraction_out:
    type: File
    outputBinding:
      glob: $(inputs.background_subtraction_out)
