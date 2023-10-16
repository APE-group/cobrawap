#!/usr/bin/env cwltool

cwlVersion: v1.2
class: Workflow

inputs:

  data: string
  pipeline_path: string
  curate_LENS_Ketamine_APE_output:
    type: string?
    default: "curate_LENS_Ketamine_APE.nix"
  curate_LENS_Ketamine_APE_sampling_rate: Any?
  curate_LENS_Ketamine_APE_spatial_scale: float
  curate_LENS_Ketamine_APE_data_name: string?
  curate_LENS_Ketamine_APE_annotations: Any?
  curate_LENS_Ketamine_APE_array_annotations: Any?
  curate_LENS_Ketamine_APE_kwargs: Any?
  curate_LENS_Ketamine_APE_t_start: Any?
  curate_LENS_Ketamine_APE_t_stop: Any?
  curate_LENS_Ketamine_APE_trial: Any?
  curate_LENS_Ketamine_APE_orientation_top: string
  curate_LENS_Ketamine_APE_orientation_right: string
  curate_LENS_Ketamine_APE_hemodynamics_correction: Any?
  curate_LENS_Ketamine_APE_path_to_reflectance_data: string?
  plot_traces_output:
    type: string?
    default: "plot.png"
  plot_traces_t_start: Any?
  plot_traces_t_stop: Any?
  plot_traces_channels: Any?

outputs:

  curate_LENS_Ketamine_APE_output:
    type: File
    outputSource: curate_LENS_Ketamine_APE/curate_LENS_Ketamine_APE_output

  plot_traces_output:
    type: File
    outputSource: plot_traces/plot_traces_output

  final_output:
    type: File
    outputSource: plot_traces/plot_traces_output

steps:

  curate_LENS_Ketamine_APE:
    run: cwl_steps/curate_LENS_Ketamine_APE.cwl
    in:
      pipeline_path: pipeline_path
      data: data
      output: curate_LENS_Ketamine_APE_output
      sampling_rate: curate_LENS_Ketamine_APE_sampling_rate
      spatial_scale: curate_LENS_Ketamine_APE_spatial_scale
      data_name: curate_LENS_Ketamine_APE_data_name
      annotations: curate_LENS_Ketamine_APE_annotations
      array_annotations: curate_LENS_Ketamine_APE_array_annotations
      kwargs: curate_LENS_Ketamine_APE_kwargs
      t_start: curate_LENS_Ketamine_APE_t_start
      t_stop: curate_LENS_Ketamine_APE_t_stop
      trial: curate_LENS_Ketamine_APE_trial
      orientation_top: curate_LENS_Ketamine_APE_orientation_top
      orientation_right: curate_LENS_Ketamine_APE_orientation_right
      hemodynamics_correction: curate_LENS_Ketamine_APE_hemodynamics_correction
      path_to_reflectance_data: curate_LENS_Ketamine_APE_path_to_reflectance_data
    out: [curate_LENS_Ketamine_APE_output]

  check_input:
    run: cwl_steps/check_input.cwl
    in:
      pipeline_path: pipeline_path
      data: curate_LENS_Ketamine_APE/curate_LENS_Ketamine_APE_output
    out: [check_input_output]

  plot_traces:
    run: cwl_steps/plot_traces.cwl
    in:
      pipeline_path: pipeline_path
      data: curate_LENS_Ketamine_APE/curate_LENS_Ketamine_APE_output
      output: plot_traces_output
      t_start: plot_traces_t_start
      t_stop: plot_traces_t_stop
      channels: plot_traces_channels
    out: [plot_traces_output]

