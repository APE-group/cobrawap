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
      location: "../scripts/plot_traces.py"
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
  t_start:
    type: Any?
    inputBinding:
      position: 3
      prefix: --t_start
  t_stop:
    type: Any?
    inputBinding:
      position: 4
      prefix: --t_stop
  channels:
    type: Any?
    inputBinding:
      position: 5
      prefix: --channels

outputs:
  plot_traces_output:
    type: File
    outputBinding:
      glob: $(inputs.output)
