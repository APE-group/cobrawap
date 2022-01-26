import argparse
import pandas as pd
from copy import deepcopy


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='+', type=str)
    CLI.add_argument("--output_img",nargs='?', type=str)
    # CLI.add_argument("--merge_key", nargs='+', type=str)
    args, unknown = CLI.parse_known_args()

    for i, datafile in enumerate(args.data):
        df = pd.read_csv(datafile)
        if i:
            full_df = full_df.merge(df, how='outer', on=None)
        else:
            full_df = deepcopy(df)
        del df

    # checking and transforming inf and nan values?

    full_df.to_html(args.output_img)

    full_df.to_csv(args.output)