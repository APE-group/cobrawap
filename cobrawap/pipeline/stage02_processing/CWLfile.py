# args mapping for stage 2

default_neo_output = lambda block, config: f"{block}.{config['NEO_FORMAT']}"

default_block = {
    'output': default_neo_output,
}

ARG_MAP = {

    'background_subtraction': {
        'output': default_neo_output,
        'output_img': lambda block, config: f"background.{config['PLOT_FORMAT']}",
        'output_array': lambda block, config: "background.npy",
    },

    'check_input': {},

    'detrending': {
        'output': default_neo_output,
        'order': lambda block, config: f"{config['DETRENDING_ORDER']}",
        'output_img_dir': lambda block, config: f"{block}_plots",
        'img_name': lambda block, config: f"{block}_trace_channel0.{config['PLOT_FORMAT']}",
    },

    'frequency_filter': default_block,

    'logMUA_estimation': {
        'output': default_neo_output,
        'output_img_dir': lambda block, config: f"{block}_plots",
        'img_name': lambda block, config: f"logMUA_trace_channel0.{config['PLOT_FORMAT']}",
        'highpass_frequency': lambda block, config: f"{config['MUA_HIGHPASS_FREQUENCY']}",
        'lowpass_frequency': lambda block, config: f"{config['MUA_LOWPASS_FREQUENCY']}",
        'logMUA_rate': lambda block, config: f"{config['logMUA_RATE']}",
    },

    'normalization': default_block,

    'phase_transform': default_block,

    'plot_power_spectrum': {
        'output_img': lambda block, config: f"power_spectrum.{config['PLOT_FORMAT']}",
    },

    'plot_processed_traces': {
        'output_img_dir': lambda block, config: f"processed_traces_{config['PLOT_TSTART']}-{config['PLOT_TSTOP']}s",
        'img_name': lambda block, config: f"processed_trace_channel0.{config['PLOT_FORMAT']}",
    },

    'roi_selection': {
        'output': default_neo_output,
        'output_img': lambda block, config: f"{block}.{config['PLOT_FORMAT']}",
    },

    'spatial_downsampling': {
        'output': default_neo_output,
        'output_img': lambda block, config: f"{block}.{config['PLOT_FORMAT']}",
    },

    'subsampling': default_block,

    'zscore': default_block,

}
