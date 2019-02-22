import argparse
import os
from autolab_core import YamlConfig

from triplet_embedding import TripletEmbedding

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Limit number of CPU cores used.
cpu_cores = [8, 9, 10, 11] # Cores (numbered 0-11)
os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

if __name__ == "__main__":

    # parse the provided configuration file, set tf settings, and train
    conf_parser = argparse.ArgumentParser(description="Train triplet embedding model")
    conf_parser.add_argument("--config", action="store", default="../cfg/train_triplet_embedding.yaml",
                               dest="conf_file", type=str, help="path to the configuration file")
    conf_args = conf_parser.parse_args()

    # read in config file information from proper section
    config = YamlConfig(conf_args.conf_file)
    model_config = config['model']
    train_config = config['train']

    model = TripletEmbedding('training', model_config)
    model.train(train_config)
