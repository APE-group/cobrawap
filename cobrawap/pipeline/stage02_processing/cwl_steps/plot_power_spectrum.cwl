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
      location: "../scripts/plot_power_spectrum.py"
    inputBinding:
      position: 0
  data:
    type: Any
    inputBinding:
      position: 1
      prefix: --data
  output:
    type: Any
    inputBinding:
      position: 2
      prefix: --output
  highpass_frequency:
    type: Any?
    inputBinding:
      position: 3
      prefix: --highpass_frequency
  lowpass_frequency:
    type: Any?
    inputBinding:
      position: 4
      prefix: --lowpass_frequency
  psd_frequency_resolution:
    type: float?
    inputBinding:
      position: 5
      prefix: --psd_frequency_resolution
  psd_overlap:
    type: float?
    inputBinding:
      position: 6
      prefix: --psd_overlap

outputs:
  plot_power_spectrum_output:
    type: File
    outputBinding:
      glob: $(inputs.output)
