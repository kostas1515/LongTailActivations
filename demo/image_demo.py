# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img_dir', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
#     result = inference_detector(model, img2process)
    # show the results
#     show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    path2save = os.path.join(args.img_dir,'../detections')
    if os.path.exists(path2save) is False:
        os.mkdir(path2save)
    for file in os.listdir(args.img_dir):
        if file.endswith(".jpg"):
            img2process = os.path.join(args.img_dir,file)
            img2save = os.path.join(path2save,file)
            result = inference_detector(model, img2process)
            model.show_result(img2process, result, out_file=img2save, score_thr = args.score_thr )


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
