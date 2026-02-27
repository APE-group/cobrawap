cwlVersion: v1.2
class: Workflow
inputs:
  curation_script_inputs:
    type:
      type: record
      name: curation_script_inputs
      fields:
      - name: data
        type:
        - File
        - Directory
      - name: output
        type: string
      - name: sampling_rate
        type: float?
      - name: spatial_scale
        type: float
      - name: data_name
        type: string?
      - name: annotations
        type: string[]?
      - name: array_annotations
        type: string[]?
      - name: kwargs
        type: string[]?
      - name: t_start
        type: float?
      - name: t_stop
        type: float?
      - name: orientation_top
        type: string
      - name: orientation_right
        type: string
  check_input_inputs:
    type:
      type: record
      name: check_input_inputs
      fields:
#      - name: data
#        type: File
      - name: output
        type: string
  plot_traces_inputs:
    type:
      type: record
      name: plot_traces_inputs
      fields:
#      - name: data
#        type: File
      - name: output_img
        type: string
      - name: plot_tstart
        type: float?
      - name: plot_tstop
        type: float?
      - name: plot_channels
        type: int[]?

outputs:
  curation_script_block_output:
    type: File
    outputSource: run_curation_script/block_output
  check_input_output:
    type: File
    outputSource: run_check_input/output
  plot_traces_output_img:
    type: File
    outputSource: run_plot_traces/output_img

steps:
  run_curation_script:
    run: /users/koehler/projects/cobrawap_ape/cobrawap/pipeline/stage01_data_entry/cwl_steps/curation_script.cwl
    in:
      data:
        source: curation_script_inputs
        valueFrom: $(self.data)
      output:
        source: curation_script_inputs
        valueFrom: $(self.output)
      sampling_rate:
        source: curation_script_inputs
        valueFrom: $(self.sampling_rate)
      spatial_scale:
        source: curation_script_inputs
        valueFrom: $(self.spatial_scale)
      data_name:
        source: curation_script_inputs
        valueFrom: $(self.data_name)
      annotations:
        source: curation_script_inputs
        valueFrom: $(self.annotations)
      array_annotations:
        source: curation_script_inputs
        valueFrom: $(self.array_annotations)
      kwargs:
        source: curation_script_inputs
        valueFrom: $(self.kwargs)
      t_start:
        source: curation_script_inputs
        valueFrom: $(self.t_start)
      t_stop:
        source: curation_script_inputs
        valueFrom: $(self.t_stop)
      orientation_top:
        source: curation_script_inputs
        valueFrom: $(self.orientation_top)
      orientation_right:
        source: curation_script_inputs
        valueFrom: $(self.orientation_right)
    out:
    - block_output
  run_check_input:
    run: /users/koehler/projects/cobrawap_ape/cobrawap/pipeline/stage01_data_entry/cwl_steps/check_input.cwl
    in:
      data: [run_curation_script/block_output]
      output:
        source: check_input_inputs
        valueFrom: $(self.output)
    out:
    - output
  run_plot_traces:
    run: /users/koehler/projects/cobrawap_ape/cobrawap/pipeline/stage01_data_entry/cwl_steps/plot_traces.cwl
    in:
      data: [run_curation_script/block_output]
#      check: [run_check_input/output]
      output_img:
        source: plot_traces_inputs
        valueFrom: $(self.output_img)
      plot_tstart:
        source: plot_traces_inputs
        valueFrom: $(self.plot_tstart)
      plot_tstop:
        source: plot_traces_inputs
        valueFrom: $(self.plot_tstop)
      plot_channels:
        source: plot_traces_inputs
        valueFrom: $(self.plot_channels)
    out:
    - output_img
requirements:
  InlineJavascriptRequirement: {}
  StepInputExpressionRequirement: {}
