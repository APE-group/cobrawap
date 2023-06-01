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
      location: "../scripts/frequency_filter.py"
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
  highpass_freq:
    type: Any?
    inputBinding:
      position: 3
      prefix: --highpass_freq
  lowpass_freq:
    type: Any?
    inputBinding:
      position: 4
      prefix: --lowpass_freq
  order:
    type: int?
    inputBinding:
      position: 5
      prefix: --order
  filter_function:
    type: string?
    inputBinding:
      position: 6
      prefix: --filter_function

outputs:
  frequency_filter_output:
    type: File
    outputBinding:
      glob: $(inputs.output)
