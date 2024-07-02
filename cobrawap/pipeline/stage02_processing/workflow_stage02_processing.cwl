#!/usr/bin/env cwltool
cwlVersion: v1.0
class: Workflow

# ------------------------------------------------------- WORKFLOW INPUTS -------------------------------------------------------

# All the inputs with their types are defined here.
# (when the parameter type ends with a question mark "?" it indicates that the parameter is optional)
inputs:
  data: File
  pipeline_path: string
  # roi selection
  roi_selection_output_img: string
  intensity_threshold: float
  crop_to_selection: boolean
  roi_selection_out: string
  # background subtraction
  background_subtraction_out: string
  background_subtraction_output_img: string

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
    run: cwl_steps/roi_selection.cwl
    in:
      data: data
      output: roi_selection_out
      output_img: roi_selection_output_img
      intensity_threshold: intensity_threshold
      crop_to_selection: crop_to_selection
      pipeline_path: pipeline_path
    out: [roi_selection_out]

  background_subtraction:
    run: cwl_steps/background_subtraction.cwl 
    in:
      data: roi_selection/roi_selection_out
      output: background_subtraction_out
      output_img: background_subtraction_output_img
      pipeline_path: pipeline_path
    out: [background_subtraction_out]
