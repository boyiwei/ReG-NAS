import argparse

from torch_geometric.graphgym.utils.agg_runs import agg_batch_proxy


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a regression model')
    parser.add_argument('--dir', dest='dir', help='Dir for batch of results',
                        required=True, type=str)
    parser.add_argument('--metric', dest='metric',
                        help='metric to select best epoch', required=False,
                        type=str, default='mse')
    return parser.parse_args()


args = parse_args()
agg_batch_proxy(args.dir, args.metric)
