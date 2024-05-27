#!/usr/bin/env python
# coding: utf-8

import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_target', type=str,
                        default='data/uspto_full/tgt-train.txt')
    parser.add_argument('--val_target', type=str,
                        default='data/uspto_full/tgt-val.txt')
    parser.add_argument('--candidate_file', type=str,
                        default='data/uspto_full/candidates.txt')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    target = set()

    with open(args.train_target, encoding='utf-8') as f:
        for line in f.readlines():
            # 去掉每行末尾的换行符，并将条目添加到集合中
            target.add(line.rstrip())

    with open(args.val_target, encoding='utf-8') as f:
        for line in f.readlines():
            target.add(line.rstrip())

    with open(args.candidate_file, 'w', encoding='utf-8') as f:
        for t in target:
            # 将集合中的每个条目写入候选文件，每个条目一行
            f.writelines(t + '\n')

    print(f'Collect {len(target)} candidates in {args.candidate_file}.')
