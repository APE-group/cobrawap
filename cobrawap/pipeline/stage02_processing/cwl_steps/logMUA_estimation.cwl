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
      location: "../scripts/logMUA_estimation.py"
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
  highpass_freq:
    type: float?
    inputBinding:
      position: 4
      prefix: --highpass_freq
  lowpass_freq:
    type: float?
    inputBinding:
      position: 5
      prefix: --lowpass_freq
  logMUA_rate:
    type: Any?
    inputBinding:
      position: 6
      prefix: --logMUA_rate
  psd_overlap:
    type: float?
    inputBinding:
      position: 7
      prefix: --psd_overlap
  fft_slice:
    type: Any?
    inputBinding:
      position: 8
      prefix: --fft_slice
  t_start:
    type: float?
    inputBinding:
      position: 9
      prefix: --t_start
  t_stop:
    type: float?
    inputBinding:
      position: 10
      prefix: --t_stop
  channels:
    type: Any?
    inputBinding:
      position: 11
      prefix: --channels

outputs:
  logMUA_estimation_output:
    type: File
    outputBinding:
      glob: $(inputs.output)
