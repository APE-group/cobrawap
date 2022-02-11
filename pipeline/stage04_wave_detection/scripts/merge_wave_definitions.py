import argparse
import numpy as np
import neo
from utils.io import load_neo, write_neo


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--waves", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--properties", nargs='?', type=lambda v: v.split(','), default=None,
                     help="paths to input data in neo format", required = False)
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                     help="path of output file")

    args = CLI.parse_args()
    waves_block = load_neo(args.waves)

    asig_names = [asig.name for asig in waves_block.segments[0].analogsignals]
    event_names = [event.name for event in waves_block.segments[0].events]

    if args.properties is None:
        args.properties = []

    for property in args.properties:
        block = load_neo(property)

        for asig in block.segments[0].analogsignals:
            if asig.name not in asig_names:
                waves_block.segments[0].analogsignals.append(asig)

        for event in block.segments[0].events:
            if event.name in event_names:
                waves_evt = waves_block.filter(name=event.name, objects="Event")[0]
                for key, value in event.annotations.items():
                    if key not in waves_evt.annotations:
                        waves_evt.annotations[key] = value
                for key, value in event.array_annotations.items():
                    if key not in waves_evt.array_annotations:
                        waves_evt.array_annotations[key] = value
            else:
                waves_block.segments[0].events.append(event)

        del block

    write_neo(args.output, waves_block)
