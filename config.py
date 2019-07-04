import yaml
import easydict
from os.path import join

import argparse
parser = argparse.ArgumentParser(description='Code for *Machine Translation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='eng-fra.yaml', help='/path/to/config/file')

args = parser.parse_args()

config_file = args.config

args = yaml.load(open(config_file))

args = easydict.EasyDict(args)

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

teacher_forcing_ratio = 0.5