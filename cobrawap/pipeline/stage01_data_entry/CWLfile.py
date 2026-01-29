# args mapping for stage 1

default_neo_output = lambda block, config: f"{block}.{config['NEO_FORMAT']}"

default_block = {
    'output': default_neo_output,
}

ARG_MAP = {

    'check_input': {
        'output': "input.check"
    },

    'enter_data_template': default_block,

    'plot_traces': {
        'output_img': lambda block, config: f"trace_{config['DATA_NAME']}.{config['PLOT_FORMAT']}",
    },

}
