from argparse import ArgumentParser

from libs import Trainer, load_config


def train():
    parser = ArgumentParser()
    parser.add_argument(
        '--exp_tag', type=str,
        default='debug',
        help='Tag string for experiment')
    parser.add_argument(
        '--run_name', type=str,
        default=None,
        help='Run name string for experiment')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Config file path')
    parser.add_argument(
        '--opts',
        type=str,
        nargs='*',
        help='Override yaml configs with the same way as detectron2')
    args = parser.parse_args()

    cfg = load_config(
        args.config,  override_opts=args.opts)
    trainer = Trainer(cfg)
    trainer.run(
        experiment_tag=args.exp_tag,
        run_name=args.run_name)


if __name__ == '__main__':

    train()
