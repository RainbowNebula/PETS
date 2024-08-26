#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
import logging
import torch
import random
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser, default_setup, launch

from ubteacher import add_teacher_config
from ubteacher.engine.trainer import DATeacherTrainer

# hacky way to register
from ubteacher.modeling.backbone.vgg16 import build_vgg16_backbone
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel, EnsembleTSModel_DA


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_teacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)


    if cfg.SFDA.Trainer == "dateacher":
        Trainer = DATeacherTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SFDA.Trainer == "dateacher":
            logger = logging.getLogger(__name__)
            if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
                setup_logger(cfg.OUTPUT_DIR, name=__name__)
            model = Trainer.build_model(cfg)
            model_teacher_static = Trainer.build_model(cfg)
            model_teacher_dynamic = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel_DA(model_teacher_static, model_teacher_dynamic, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            logger.info("".center(100, "#"))
            logger.info("The result of dynamic_Teacher".center(100, "#"))
            logger.info("".center(100, "#"))
            res1 = Trainer.test(cfg, ensem_ts_model.modelTeacher_dynamic)
            logger.info("".center(100, "#"))
            logger.info("The result of static_Teacher".center(100, "#"))
            logger.info("".center(100, "#"))
            res2 = Trainer.test(cfg, ensem_ts_model.modelTeacher_static)
            return res1, res2
        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
