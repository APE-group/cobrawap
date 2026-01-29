# args mapping for stage 3

default_neo_output = lambda block, config: f"{block}.{config['NEO_FORMAT']}"

default_block = {
    'output': default_neo_output,
}

ARG_MAP = {

    'hilbert_phase': {
        'output': default_neo_output,
        'output_img_dir': lambda block, config: f"{block}_plots",
        'img_name': lambda block, config: f"{block}_channel0.{config['PLOT_FORMAT']}",
    },

    'check_input': {
        'output': "input.check"
    },

}
