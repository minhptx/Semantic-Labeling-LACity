#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import shutil
import ujson
from pathlib import Path

import numpy as np

# import gmtk.config
# gmtk.config.USE_C_EXTENSION = False
from experiments.arg_helper import parse_kfold, str2bool
from semantic_labeling import create_semantic_typer
from semantic_modeling.assembling.learning.evaluate import predict_sm, evaluate
from datetime import datetime

from semantic_modeling.assembling.learning.online_learning import create_default_model, online_learning
from semantic_modeling.assembling.learning.shared_models import TrainingArgs
from semantic_modeling.assembling.undirected_graphical_model.model import Model
from semantic_modeling.config import config
from semantic_modeling.data_io import get_semantic_models, get_short_train_name
from semantic_modeling.settings import Settings
from semantic_modeling.utilities.serializable import serializeCSV, serializeJSON


def get_shell_args():
    parser = argparse.ArgumentParser('Assembling experiment')
    parser.register("type", "boolean", str2bool)

    # copied from settings.py
    parser.add_argument('--random_seed', type=int, default=120, help='default 120')
    parser.add_argument('--n_samples', type=int, default=1000, help='default 1000')

    parser.add_argument('--semantic_labeling_method', type=str, default="ReImplMinhISWC", help='can be ReImplMinhISWC and MohsenISWC, default ReImplMinhISWC')
    parser.add_argument('--semantic_labeling_top_n_stypes', type=int, default=4, help='Default is top 4')
    
    parser.add_argument('--auto_labeling_method', type=str, default='AUTO_LBL_MAX_F1', help='can be AUTO_LBL_MAX_F1 and AUTO_LBL_PRESERVED_STRUCTURE (default AUTO_LBL_MAX_F1)')
    
    parser.add_argument('--data_constraint_guess_datetime_threshold', type=int, default=0.5, help='default 0.5')
    parser.add_argument('--data_constraint_valid_threshold', type=int, default=0.95, help='default is 0.95')
    parser.add_argument('--data_constraint_n_comparison_samples', type=int, default=150, help='default is 150')

    parser.add_argument('--training_beam_width', type=int, default=10, help='default 10')
    parser.add_argument('--searching_beam_width', type=int, default=10, help='default 10')
    parser.add_argument('--searching_max_data_node_hop', type=int, default=2, help='default 2')
    parser.add_argument('--searching_max_class_node_hop', type=int, default=2, help='default 2')
    parser.add_argument('--searching_n_explore_result', type=int, default=5, help='default 5')
    parser.add_argument('--searching_triple_adviser_max_candidate', type=int, default=15, help='default 15')
    parser.add_argument('--searching_early_stopping_method', type=str, default='NoEarlyStopping', help='can be NoEarlyStopping or MinProb (default NoEarlyStopping)')
    parser.add_argument('--searching_early_stopping_minimum_expected_accuracy', type=int, default=0, help='default 0')
    parser.add_argument('--searching_early_stopping_min_prob_args', type=str, default="[0.01]", help='default is [0.01]')

    parser.add_argument('--parallel_gmtk_n_threads', type=int, default=8, help='default is 8 threads')
    parser.add_argument('--parallel_n_process', type=int, default=4, help='default is 4 processes')
    parser.add_argument('--parallel_n_annotators', type=int, default=8, help='default is 8')
    parser.add_argument('--max_n_tasks', type=int, default=80, help='default is 80')

    # copied from shared_models.py/TrainingArgs
    parser.add_argument('--n_epoch', type=int, default=40, help='default 40')
    parser.add_argument('--n_switch', type=int, default=10, help='default 10')
    parser.add_argument('--n_iter_eval', type=int, default=5, help='default 5')
    parser.add_argument('--mini_batch_size', type=int, default=200, help='default 200')
    parser.add_argument('--shuffle_mini_batch', type="boolean", default=False, help='Default false')
    parser.add_argument('--manual_seed', type=int, default=120, help='default 120')
    parser.add_argument('--report_final_loss', type="boolean", default=True, help='Default false')
    parser.add_argument('--optparams', type=str, default=ujson.dumps(dict(lr=0.1)), help='default dict(lr=0.1)')
    parser.add_argument('--optimizer', type=str, default='ADAM', help='default ADAM (when using ADAM, u must use amsgrad')
    parser.add_argument('--parallel', type="boolean", default=True, help="Default true")

    # custom arguments
    parser.add_argument('--dataset', type=str, default=None, help="Dataset name")
    parser.add_argument('--kfold', type=str, default='{"train_sm_ids": ["s01-s03"], "test_sm_ids": ["s02-s02"]}', help='kfold json object of {train_sm_ids: [], test_sm_ids: []}')
    parser.add_argument('--n_iter', type=int, default=2, help="Number of iteration for online learning. Default is 2")
    parser.add_argument('--exp_dir', type=str, help='Experiment directory, must be existed before')

    args = parser.parse_args()

    try:
        args.searching_early_stopping_min_prob_args = ujson.loads(args.searching_early_stopping_min_prob_args)
        args.optparams = ujson.loads(args.optparams)
        assert args.dataset is not None
        assert args.semantic_labeling_method in {Settings.ReImplMinhISWC, Settings.MohsenISWC, Settings.OracleSL}
        assert args.auto_labeling_method in {Settings.ALGO_AUTO_LBL_MAX_F1, Settings.ALGO_AUTO_LBL_PRESERVED_STRUCTURE}
        assert args.searching_early_stopping_method in {Settings.ALGO_ES_DISABLE, Settings.ALGO_ES_MIN_PROB}

        args.kfold = parse_kfold(args.dataset, args.kfold)
    except AssertionError:
        parser.print_help()
        raise

    # update args to settings
    settings = Settings.get_instance(False)
    settings.random_seed = args.random_seed
    settings.n_samples = args.n_samples
    settings.semantic_labeling_method = args.semantic_labeling_method
    settings.semantic_labeling_top_n_stypes, = args.semantic_labeling_top_n_stypes,
    settings.auto_labeling_method = args.auto_labeling_method
    settings.data_constraint_guess_datetime_threshold = args.data_constraint_guess_datetime_threshold
    settings.data_constraint_valid_threshold = args.data_constraint_valid_threshold
    settings.data_constraint_n_comparison_samples, = args.data_constraint_n_comparison_samples,
    settings.training_beam_width = args.training_beam_width
    settings.searching_beam_width = args.searching_beam_width
    settings.searching_max_data_node_hop = args.searching_max_data_node_hop
    settings.searching_max_class_node_hop = args.searching_max_class_node_hop
    settings.searching_n_explore_result, = args.searching_n_explore_result,
    settings.searching_triple_adviser_max_candidate = args.searching_triple_adviser_max_candidate
    settings.searching_early_stopping_method = args.searching_early_stopping_method
    settings.searching_early_stopping_minimum_expected_accuracy = args.searching_early_stopping_minimum_expected_accuracy
    settings.searching_early_stopping_min_prob_args, = args.searching_early_stopping_min_prob_args,
    settings.parallel_gmtk_n_threads = args.parallel_gmtk_n_threads
    settings.parallel_n_process = args.parallel_n_process
    settings.parallel_n_annotators = args.parallel_n_annotators
    settings.max_n_tasks = args.max_n_tasks

    settings.log_current_settings()

    train_args = TrainingArgs(args.n_epoch, args.n_switch, args.mini_batch_size,
                     args.shuffle_mini_batch, args.manual_seed, args.report_final_loss,
                     args.optparams, args.optimizer, args.n_iter_eval, args.parallel)

    return args, settings, train_args

if __name__ == '__main__':
    args, settings, training_args = get_shell_args()

    dataset = args.dataset
    source_models = {sm.id: sm for sm in get_semantic_models(dataset)}

    exp_dir = Path(args.exp_dir)
    assert exp_dir.exists()

    train_sms = [source_models[sid] for sid in args.kfold['train_sm_ids']]
    test_sms = [source_models[sid] for sid in args.kfold['test_sm_ids']]

    create_semantic_typer(dataset, train_sms).semantic_labeling(
        train_sms,
        test_sms,
        top_n=settings.semantic_labeling_top_n_stypes,
        eval_train=True)

    workdir = Path(config.fsys.debug.as_path()) / dataset / "main_experiments" / get_short_train_name(train_sms)
    workdir.mkdir(exist_ok=True, parents=True)

    # training
    assert not Path(workdir / "models" / f"exp_no_0").exists()
    model = create_default_model(dataset, train_sms, training_args, workdir / "models")
    # model = Model.from_file(dataset, workdir / "models" / "exp_no_0")
    model = online_learning(model, dataset, train_sms, train_sms, workdir, training_args, iter_range=(1, args.n_iter+1))  # 2 iter then range is (1, 3), we shift by one
    model_dir = Path(workdir / "models" / f"exp_no_{args.n_iter}")
    assert model_dir.exists()

    # evaluate
    predictions = predict_sm(model, dataset, train_sms, test_sms, workdir=model_dir)
    kfold_eval = evaluate(test_sms, predictions, model_dir)

    # backup result
    shutil.move(workdir, exp_dir / get_short_train_name(train_sms))

    serializeCSV(kfold_eval, exp_dir / f"kfold-{get_short_train_name(train_sms)}.test.csv")
    serializeJSON(args, exp_dir / f"kfold-{get_short_train_name(train_sms)}.meta.txt", indent=4)
