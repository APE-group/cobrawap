#!/usr/bin/env cwltool
cwlVersion: v1.0
class: Workflow

# ------------------------------------------------------- WORKFLOW INPUTS -------------------------------------------------------

# All the inputs with their types are defined here.
# (when the parameter type ends with a question mark "?" it indicates that the parameter is optional)
inputs:
  data: string
  output_img: string
  pipeline_path: string
  intensity_threshold: float
  crop_to_selection: boolean
  roi_selection_out: string
  background_subtraction_out: string

outputs:
  step1:
    type: File
    outputSource: roi_selection/roi_selection_out
  step2:
    type: File
    outputSource: background_subtraction/background_subtraction_out
  final_output:
    type: File
    outputSource: background_subtraction/background_subtraction_out


steps:
  
  roi_selection:
    run: cwl_steps/roi_selection_wf.cwl
    in:
      data: data
      output_img: output_img
      roi_selection_out: roi_selection_out
      pipeline_path: pipeline_path
      intensity_threshold: intensity_threshold
      crop_to_selection: crop_to_selection
    out: [roi_selection_out]

  background_subtraction:
    run: cwl_steps/background_subtraction_wf.cwl 
    in:
      pipeline_path: pipeline_path
      input_file: roi_selection/roi_selection_out
      background_subtraction_out: background_subtraction_out
    out: [background_subtraction_out]
