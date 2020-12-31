from fusion_composer import *
from helper import duplicate_fusion_logs
import argparse

def get_options():
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-l", "--logfile", type=str, required=True, help="The path to the logfile")
    parser.add_argument('-p','--post_ops', nargs='+', required=True, help="Post ops to each layer")
    options = parser.parse_args()
    return options

if __name__ == '__main__':
    options = get_options()
    duplicate_fusion_logs(options.logfile, options.post_ops)