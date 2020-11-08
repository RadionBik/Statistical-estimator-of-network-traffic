
import argparse
import dataclasses
import os
import pathlib
from pprint import pprint

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data.dataloader import DataLoader

import features.evaluation
import settings
from features.data_utils import load_train_test_dataset, quantize_datatset, restore_features
from features.evaluation import evaluate_traffic
from features.gaussian_quantizer import GaussianQuantizer
from nn_generators.nn_utils import generate_states, get_model_size
from nn_generators.scenario_mapper import SCENARIO_MAPPER

seed_everything(settings.RANDOM_SEED)
CURR_DIR = settings.BASE_DIR / 'nn_generators'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help='path to preprocessed .csv dataset',
        required=True
    )

    parser.add_argument(
        '--quantizer_path',
        help='path to the quantizer checkpoint',
        required=True
    )
    parser.add_argument(
        '--generator_name',
        dest='generator_name',
        help='must be either TCN or RNN',
        default='RNN'
    )
    parser.add_argument(
        '--log_neptune',
        dest='log_neptune',
        action='store_true',
        default=False
    )
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    train_df, test_df = load_train_test_dataset(args.dataset)

    quantizer = GaussianQuantizer.from_pretrained(args.quantizer_path)
    try:
        class_mapper = SCENARIO_MAPPER[args.generator_name]
    except KeyError:
        raise ValueError('cannot find such generator')

    model_config = class_mapper.config(
        scenario=pathlib.Path(args.quantizer_path).stem,
        n_classes=quantizer.n_tokens,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_states, test_states = quantize_datatset(quantizer, train_df, test_df, model_config.window_size)

    train_size = len(train_states) - int(len(train_states) * model_config.val_size)

    train_states, val_states = train_states[:train_size], train_states[train_size:]

    train_dataset = class_mapper.dataset(train_states, window=model_config.window_size)
    val_dataset = class_mapper.dataset(val_states, window=model_config.window_size)

    pprint(model_config)

    model = class_mapper.model(model_config)

    model_config.model_size = get_model_size(model)
    cpu_counter = os.cpu_count()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=model_config.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=cpu_counter,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=model_config.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=cpu_counter,
    )

    neptune_logger = NeptuneLogger(
        api_key=settings.NEPTUNE_API_TOKEN,
        offline_mode=not args.log_neptune,
        close_after_fit=False,
        project_name=settings.NEPTUNE_PROJECT,
        experiment_name=args.generator_name,
        params=dataclasses.asdict(model_config),
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.005,
        patience=model_config.es_patience,
        verbose=False,
        mode='min'
    )

    trainer = Trainer(
        callbacks=[early_stop_callback],
        auto_lr_find=False,
        gpus=int(device == 'cuda'),
        gradient_clip_val=model_config.grad_clip,
        logger=neptune_logger,
        deterministic=True,
        check_val_every_n_epoch=1,
        precision=16 if device == 'cuda' else 32,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    # evaluation part
    gen_states = generate_states(model,
                                 sample_number=len(test_states),
                                 window_size=model_config.window_size,
                                 device=device)

    state_metrics = features.evaluation.calc_stats(test_states, gen_states)

    gen_df = restore_features(quantizer, gen_states)
    packet_metrics = evaluate_traffic(gen_df, test_df)

    for k, v in dict(**state_metrics, **packet_metrics).items():
        neptune_logger.experiment.log_metric(k, v)

    neptune_logger.experiment.stop()


if __name__ == '__main__':
    main()
