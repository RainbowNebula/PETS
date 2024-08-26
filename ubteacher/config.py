# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_teacher_config(cfg):
    """
    Add config for SFDA.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True
    _C.SOLVER.GD_LR = 0.001  # learning rate of domain classifier

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"
    _C.MODEL.LABLED_WEIGHTS = "/mnt/csip-101/Source-Only/output/baseline/K_car/model_final.pth"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)
    _C.SOLVER.EPOCHS = 10
    _C.SOLVER.Period = 1
    _C.SOLVER.WARM_UP_EPOCH = 2

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = False
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SFDA = CN()

    # Semi-supervised training
    _C.SFDA.Trainer = "ubteacher"
    _C.SFDA.BBOX_THRESHOLD = 0.7
    _C.SFDA.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SFDA.TEACHER_UPDATE_ITER = 1
    _C.SFDA.BURN_UP_STEP = 12000
    _C.SFDA.EMA_KEEP_RATE_dynamic = 0.996
    _C.SFDA.UNSUP_LOSS_WEIGHT = 4.0
    _C.SFDA.UNSUP_RPN_LOSS_WEIGHT = 0.0
    _C.SFDA.UNSUP_ROI_LOSS_WEIGHT = 0.0
    _C.SFDA.SUP_LOSS_WEIGHT = 0.5
    _C.SFDA.LOSS_WEIGHT_TYPE = "standard"
    _C.SFDA.PSEUDO_THRESHOLD = 0.5
    _C.SFDA.EMA_START = 0

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True
