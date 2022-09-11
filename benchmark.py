# Copyright (c) OpenMMLab. All rights reserved.
import time
import argparse
import os
import sys
from pathlib import Path
import torch
from models.common import DetectMultiBackend
from utils.datasets import create_dataloader
from utils.general import (check_dataset, check_yaml, non_max_suppression)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument(
        '--repeat-num',
        type=int,
        default=1,
        help='number of repeat times of measurement for averaging the results')
    parser.add_argument(
        '--max-iter', type=int, default=2000, help='num of max iter')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    opt = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(opt.local_rank)

    opt.data = check_yaml(opt.data)  # check YAML
    return opt


def measure_inference_speed(opt):
    log_interval = opt.log_interval
    max_iter = opt.max_iter

    device = torch.device('cuda:0')
    model = DetectMultiBackend(opt.weights, device=device)
    data = check_dataset(opt.data)
    model.eval()

    data_loader = create_dataloader(data['val'],
                                    640,
                                    batch_size=1,
                                    stride=32,
                                    single_cls=False,
                                    pad=0.5,
                                    rect=False,
                                    workers=0)[0]

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i, data in enumerate(data_loader):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        img = data[0].to(device, non_blocking=True)
        img = img.float()
        img /= 255  # 0 - 255 to 0.0 - 1.0

        with torch.no_grad():
            out, _ = model(img, augment=False, val=True)
            non_max_suppression(out, 0.001, 0.65, multi_label=True)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps


def repeat_measure_inference_speed(opt):
    assert opt.repeat_num >= 1

    fps_list = []

    for _ in range(opt.repeat_num):
        fps_list.append(measure_inference_speed(opt))

    if opt.repeat_num > 1:
        fps_list_ = [round(fps, 1) for fps in fps_list]
        times_pre_image_list_ = [round(1000 / fps, 1) for fps in fps_list]
        mean_fps_ = sum(fps_list_) / len(fps_list_)
        mean_times_pre_image_ = sum(times_pre_image_list_) / len(
            times_pre_image_list_)
        print(
            f'Overall fps: {fps_list_}[{mean_fps_:.1f}] img / s, '
            f'times per image: '
            f'{times_pre_image_list_}[{mean_times_pre_image_:.1f}] ms / img',
            flush=True)
        return fps_list

    return fps_list[0]


def main():
    opt = parse_args()
    repeat_measure_inference_speed(opt)


if __name__ == '__main__':
    main()
