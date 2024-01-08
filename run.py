# -*- coding: utf-8 -*-
import json
import argparse

from src.wikitables import wkt_train
from src.webquerytable import wqt_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='select dataset')
    args = parser.parse_args()

    # 5-fold cross validation on wikitables
    if args.dataset == 'wikitables':
        config = json.load(open("configs/wikitables.json"))
        wkt_train(config)
    # train and test on webquerytable
    elif args.dataset == 'webquerytable':
        config = json.load(open("configs/webquerytable.json"))
        wqt_train(config)

