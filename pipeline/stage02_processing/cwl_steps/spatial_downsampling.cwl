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
      location: "../scripts/spatial_downsampling.py"
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
  macro_pixel_dim:
    type: int?
    inputBinding:
      position: 4
      prefix: --macro_pixel_dim

outputs:
  spatial_downsampling_output:
    type: File
    outputBinding:
      glob: $(inputs.output)
