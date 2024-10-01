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
      location: "../scripts/roi_selection.py"
    inputBinding:
      position: 0
  data:
    type: string
    inputBinding:
      position: 1
      prefix: --data
  roi_selection_out:
    type: string
    inputBinding:
      position: 2
      prefix: --output
  output_img:
    type: string?
    inputBinding:
      position: 3
      prefix: --output_img
  intensity_threshold:
    type: float?
    inputBinding:
      position: 4
      prefix: --intensity_threshold
  crop_to_selection:
    type: boolean?
    inputBinding:
      position: 5
      prefix: --crop_to_selection

#stdout:
 # $(inputs.output)

#outputs:
#  roi_selection_out:
#    type: stdout
outputs:
  roi_selection_out:
    type: File
    outputBinding:
      glob: $(inputs.roi_selection_out)