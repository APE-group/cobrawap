#!/usr/bin/env cwltool

cwlVersion: v1.2
class: Workflow

inputs:

  data: Any
  pipeline_path: string
  plot_power_spectrum_output:
    type: string?
    default: "plot.png"
  plot_power_spectrum_highpass_frequency: Any?
  plot_power_spectrum_lowpass_frequency: Any?
  plot_power_spectrum_psd_frequency_resolution: float?
  plot_power_spectrum_psd_overlap: float?
  plot_processed_trace_original_data: Any
  plot_processed_trace_img_dir: Any
  plot_processed_trace_img_name: string?
  plot_processed_trace_t_start: Any?
  plot_processed_trace_t_stop: Any?
  plot_processed_trace_channels: int?

outputs:

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

  plot_power_spectrum:
    run: cwl_steps/plot_power_spectrum.cwl
    in:
      pipeline_path: pipeline_path
      data: data
      output: plot_power_spectrum_output
      highpass_frequency: plot_power_spectrum_highpass_frequency
      lowpass_frequency: plot_power_spectrum_lowpass_frequency
      psd_frequency_resolution: plot_power_spectrum_psd_frequency_resolution
      psd_overlap: plot_power_spectrum_psd_overlap
    out: [plot_power_spectrum_output]

  plot_processed_trace:
    run: cwl_steps/plot_processed_trace.cwl
    in:
      pipeline_path: pipeline_path
      data: plot_power_spectrum/plot_power_spectrum_output
      original_data: plot_processed_trace_original_data
      img_dir: plot_processed_trace_img_dir
      img_name: plot_processed_trace_img_name
      t_start: plot_processed_trace_t_start
      t_stop: plot_processed_trace_t_stop
      channels: plot_processed_trace_channels
    out: [plot_processed_trace_output]

