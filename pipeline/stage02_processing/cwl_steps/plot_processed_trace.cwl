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
      location: "../scripts/plot_processed_trace.py"
    inputBinding:
      position: 0
  original_data:
    type: string
    inputBinding:
      position: 1
      prefix: --original_data
  processed_data:
    type: string
    inputBinding:
      position: 2
      prefix: --processed_data
  img_dir:
    type: string
    inputBinding:
      position: 3
      prefix: --img_dir
  img_name:
    type: string?
    inputBinding:
      position: 4
      prefix: --img_name
  t_start:
    type: float?
    inputBinding:
      position: 5
      prefix: --t_start
  t_stop:
    type: float?
    inputBinding:
      position: 6
      prefix: --t_stop
  channels:
    type: int?
    inputBinding:
      position: 7
      prefix: --channels

outputs:
  plot_processed_trace_output:
    type: stdout
