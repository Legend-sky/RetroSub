import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

import GPUtil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dir', type=str)
    parser.add_argument('--mem', type=float, default=0.5)
    args = parser.parse_args()
    return args


# python -u data_utils/dispatch_infer.py --model uspto_full_retrosub --dir subextraction
if __name__ == "__main__":  #执行分布式的推理任务 通过多进程并行的方式来处理数据分块（chunks），并利用多个GPU资源来加速推理过程
    args = parse_args()
    model_name = args.model
    test_data_dir_name = args.dir
    print(datetime.now())
    total_chunks = 200
    test_data_dir = f'./data/uspto_full/{test_data_dir_name}'

    log_dir = f'logs/{model_name}_{test_data_dir_name}'
    result_dir = f'data/result_{model_name}_{test_data_dir_name}'
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    #分块处理和GPU资源分配
    pro_list = []   #初始化一个进程列表
    for chunk_id in range(total_chunks):
        device_id = GPUtil.getFirstAvailable(
            order='memory', maxLoad=0.8, maxMemory=args.mem, attempts=100, interval=60, verbose=False)[0]   #获取一个符合条件的可用GPU设备ID
        log_file = f'{log_dir}/{chunk_id}_{total_chunks}.log'
        #构建bash命令，用于执行分块预测脚本（step4_predict_chunk.sh），传递模型名称、分块ID、总分块数、测试数据目录、设备ID和结果目录等参数
        bash_command = f'bash scripts/uspto_full/step4_predict_chunk.sh  {model_name} {chunk_id} {total_chunks} {test_data_dir} {device_id} {result_dir}'
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                bash_command, shell=True, stdout=f, stderr=f)
            print(datetime.now(), process.args)
            pro_list.append(process)
        # wait some time for the model to be loaded to GPU
        time.sleep(60)

    for process in pro_list:
        process.wait()
        if process.returncode != 0:
            print(process.args)
