#!/usr/bin/env cwltool

cwlVersion: v1.2
class: Workflow

inputs:

  data: File
  pipeline_path: string
  roi_selection_output: string
  roi_selection_output_img: Any?
  roi_selection_intensity_threshold: float?
  roi_selection_crop_to_selection: Any?
  background_subtraction_output: string
  background_subtraction_output_img: Any?
  background_subtraction_output_array: Any?
  normalization_output: string
  normalization_normalize_by: string?
  plot_power_spectrum_output: string
  plot_power_spectrum_highpass_freq: Any?
  plot_power_spectrum_lowpass_freq: Any?
  plot_power_spectrum_psd_freq_res: float?
  plot_power_spectrum_psd_overlap: float?
  plot_processed_trace_original_data: string
  plot_processed_trace_processed_data: string
  plot_processed_trace_img_dir: string
  plot_processed_trace_img_name: string?
  plot_processed_trace_t_start: float?
  plot_processed_trace_t_stop: float?
  plot_processed_trace_channels: int?

outputs:

  roi_selection_output:
    type: File
    outputSource: roi_selection/roi_selection_output

  background_subtraction_output:
    type: File
    outputSource: background_subtraction/background_subtraction_output

  normalization_output:
    type: File
    outputSource: normalization/normalization_output

  plot_power_spectrum_output:
    type: File
    outputSource: plot_power_spectrum/plot_power_spectrum_output

  final_output:
    type: File
    outputSource: plot_power_spectrum/plot_power_spectrum_output

steps:

  check_input:
    run: cwl_steps/check_input.cwl
    in:
      pipeline_path: pipeline_path
      data: data
    out: [check_input_output]

  roi_selection:
    run: cwl_steps/roi_selection.cwl
    in:
      pipeline_path: pipeline_path
      data: data
      output: roi_selection_output
      output_img: roi_selection_output_img
      intensity_threshold: roi_selection_intensity_threshold
      crop_to_selection: roi_selection_crop_to_selection
    out: [roi_selection_output]

  background_subtraction:
    run: cwl_steps/background_subtraction.cwl
    in:
      pipeline_path: pipeline_path
      data: roi_selection/roi_selection_output
      output: background_subtraction_output
      output_img: background_subtraction_output_img
      output_array: background_subtraction_output_array
    out: [background_subtraction_output]

  normalization:
    run: cwl_steps/normalization.cwl
    in:
      pipeline_path: pipeline_path
      data: background_subtraction/background_subtraction_output
      output: normalization_output
      normalize_by: normalization_normalize_by
    out: [normalization_output]

  plot_power_spectrum:
    run: cwl_steps/plot_power_spectrum.cwl
    in:
      pipeline_path: pipeline_path
      data: normalization/normalization_output
      output: plot_power_spectrum_output
      highpass_freq: plot_power_spectrum_highpass_freq
      lowpass_freq: plot_power_spectrum_lowpass_freq
      psd_freq_res: plot_power_spectrum_psd_freq_res
      psd_overlap: plot_power_spectrum_psd_overlap
    out: [plot_power_spectrum_output]

  plot_processed_trace:
    run: cwl_steps/plot_processed_trace.cwl
    in:
      pipeline_path: pipeline_path
      data: plot_power_spectrum/plot_power_spectrum_output
      original_data: plot_processed_trace_original_data
      processed_data: plot_processed_trace_processed_data
      img_dir: plot_processed_trace_img_dir
      img_name: plot_processed_trace_img_name
      t_start: plot_processed_trace_t_start
      t_stop: plot_processed_trace_t_stop
      channels: plot_processed_trace_channels
    out: [plot_processed_trace_output]

