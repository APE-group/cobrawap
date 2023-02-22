#!/usr/bin/env cwltool
cwlVersion: v1.0
class: Workflow

# ------------------------------------------------------- WORKFLOW INPUTS -------------------------------------------------------

# All the inputs with their types are defined here.
# (when the parameter type ends with a question mark "?" it indicates that the parameter is optional)
inputs:
  data: string
  output_img: string
  output: string
  pipeline_path: string
  intensity_threshold: float
  crop_to_selection: boolean
  background_subtraction_output: string

outputs:
  output_file:
    type: File
    outputSource: background_subtraction/background_subtraction_out


steps:
  roi_selection:
    run: cwl_steps/roi_selection_wf.cwl
    in:
      data: data
      output_img: output_img
      output: output
      pipeline_path: pipeline_path
      intensity_threshold: intensity_threshold
      crop_to_selection: crop_to_selection
    out: [roi_selection_out]

  background_subtraction:
    run: cwl_steps/background_subtraction_wf.cwl 
    in:
      pipeline_path: pipeline_path
      input_file: roi_selection/roi_selection_out
      output_file: background_subtraction_output
    out: [background_subtraction_out]
