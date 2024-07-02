"""
Merge pandas DataFrames based on the values of selected columns.
"""

import argparse
from pathlib import Path
import pandas as pd
from copy import deepcopy
from utils.parse import none_or_str

CLI = argparse.ArgumentParser()
CLI.add_argument("--data", nargs='+', type=str, required=True,
                 help="path to input data in neo format")
CLI.add_argument("--output", nargs='?', type=Path, required=True,
                 help="path of output file")
CLI.add_argument("--output_img", nargs='?', type=none_or_str, default=None,
                 help="path of output image file")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    for i, datafile in enumerate(args.data):
        df = pd.read_csv(datafile)
        df.drop(df.columns[df.columns.str.contains('unnamed', case=False)],
                axis=1, inplace=True)
        if i:
            full_df = full_df.merge(df, how='outer', on='wavefronts_id')
        else:
            full_df = deepcopy(df)
        del df

    full_df.to_html(args.output_img)

    full_df.to_csv(args.output)
