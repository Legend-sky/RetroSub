import argparse
import random, os
import logging
import numpy as np
import torch
import json

from mips import MIPS, augment_query, l2_to_ip
from retriever import ProjEncoder, DataLoader 
from utils import move_to_device, asynchronous_load
from data import Vocab, BOS, EOS

logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--allow_hit', action='store_true')

    parser.add_argument('--vocab_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--args_path', type=str)
    parser.add_argument('--batch_size', type=int, default=2048)

    parser.add_argument('--nprobe', type=int, default=64)
    parser.add_argument('--index_file', type=str)
    parser.add_argument('--index_path', type=str)
    
    
    return parser.parse_args()

def main(args):
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
    logger.info('Loading model...')
    # device = torch.device('cuda', 0)
    ## device = torch.device('cpu')
    
    vocab = Vocab(args.vocab_path, 0, [BOS, EOS])   #加载保存src所有token的字典文件
    model_args = torch.load(args.args_path)         #模型参数
    model = ProjEncoder.from_pretrained(vocab, model_args, args.ckpt_path)  #根据参数加载预训练模型
    # model.to(device)
    
    logger.info('Collecting data...')
    
    data_r = [] #候选集candidates
    with open(args.index_file) as f:
        for line in f.readlines():
            r = line.strip()
            data_r.append(r)

    data_q = []     #产物
    data_qr = []    #反应物集合
    with open(args.input_file, 'r') as f:
        for line in f.readlines():
            q, r = line.strip().split('\t')
            data_q.append(q)
            data_qr.append(r)

    logger.info('Collected %d instances', len(data_q))
    textq, textqr, textr = data_q, data_qr, data_r
    data_loader = DataLoader(data_q, vocab, args.batch_size) 
    
    #MIPS是一种 Faiss 中的近似最近邻搜索算法。 
    #Faiss 是一个用于高维向量索引和相似度搜索的库，而 MIPS 类则是在 Faiss 中提供了一种简单的方式来管理和使用索引对象。
    mips = MIPS.from_built(args.index_path, nprobe=args.nprobe) #args.nprobe=64 
    
    max_norm = torch.load(os.path.dirname(args.index_path)+'/max_norm.pt')
    # mips.to_gpu() 
    # model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.eval()


    logger.info('Start search')
    cur, tot = 0, len(data_q)
    with open(args.output_file, 'w') as fo:
        for batch in asynchronous_load(data_loader):
            with torch.no_grad():
                q = torch.from_numpy(batch).contiguous().t() 
                #q = move_to_device(batch, torch.device('cpu')).t()
                bsz = q.size(0)
                vecsq = model(q, batch_first=True).detach().cpu().numpy()
            vecsq = augment_query(vecsq)
            D, I = mips.search(vecsq, args.topk+1)
            D = l2_to_ip(D, vecsq, max_norm) / (max_norm * max_norm)
            for i, (Ii, Di) in enumerate(zip(I, D)):
                item = [textq[cur+i], textqr[cur+i]]
                for pred, s in zip(Ii, Di):
                    if args.allow_hit or textr[pred] != textqr[cur+i]:
                        item.append(textr[pred])
                        item.append(str(float(s)))
                item = item[:2+2*args.topk]
                assert len(item) == 2+2*args.topk
                fo.write('\t'.join(item)+'\n')
            cur += bsz
            logger.info('finished %d / %d', cur, tot)
              

if __name__ == "__main__":
    args = parse_args()
    main(args)
