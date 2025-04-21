"""
Train a diffusion model on images.
"""

import argparse
import time

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets_spatial import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_spatial_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure('spatial_train_result/'+args.data_type)

    logger.log("creating model and diffusion...")
    model, diffusion = create_spatial_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_type=args.data_type,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rank=args.rank
    )

    logger.log("training...")
    start_time = time.time()
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

    end_time = time.time()
    training_time_minutes = (end_time - start_time) / 60
    print("Spatial Training Time：{:.2f} m".format(training_time_minutes))


def create_argparser():
    data_idx = 0
    type_list = ['pavia', 'chikusei', 'ksc', 'houston']
    defaults = dict(
        data_type=type_list[data_idx],
        data_dir="datasets",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=30000,
        batch_size=16,  # because we only have 5*5 samples
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
