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
      location: "../scripts/hierarchical_spatial_sampling.py"
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
  n_bad_nodes:
    type: Any?
    inputBinding:
      position: 4
      prefix: --n_bad_nodes
  exit_condition:
    type: Any?
    inputBinding:
      position: 5
      prefix: --exit_condition
  signal_eval_method:
    type: Any?
    inputBinding:
      position: 6
      prefix: --signal_eval_method
  voting_threshold:
    type: Any?
    inputBinding:
      position: 7
      prefix: --voting_threshold
  output_array:
    type: Any?
    inputBinding:
      position: 8
      prefix: --output_array

outputs:
  hierarchical_spatial_sampling_output:
    type: File
    outputBinding:
      glob: $(inputs.output)
