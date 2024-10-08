# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

from .coco_evaluation import COCOEvaluatorNEW
from .pascal_evaluation import PascalVOCDetectionEvaluatorNEW

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

from ubteacher.data.build import (
    build_detection_test_loader,
    build_detection_da_train_loader_two_crops
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel_DA
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.solver.build import build_lr_scheduler
from .boxes import box_iou
from .wbf import weighted_boxes_fusion
import time
import datetime


class DATeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
                cfg (CfgNode):
            Use the custom checkpointer, which loads other backbone models
            with matching heuristics.
        """
        logger = logging.getLogger(__name__)
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger(cfg.OUTPUT_DIR, name=__name__)
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        self.start_iter = 0
        self.max_iter = len(data_loader.dataset.dataset) // cfg.SOLVER.IMG_PER_BATCH_UNLABEL + 1
        self.epochs = cfg.SOLVER.EPOCHS
        cfg.SOLVER.STEPS = (self.max_iter * cfg.SOLVER.WARM_UP_EPOCH, )
        cfg.TEST.EVAL_PERIOD = self.max_iter

        # create a student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create a teacher model
        model_teacher_static = self.build_model(cfg)
        self.model_teacher_static = model_teacher_static
        model_teacher_dynamic = self.build_model(cfg)
        self.model_teacher_dynamic = model_teacher_dynamic

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )


        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        ensem_ts_model = EnsembleTSModel_DA(model_teacher_static,
                                            model_teacher_dynamic,
                                            model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
        )
        self.cfg = cfg

        self.EMA_KEEP_RATE_dynamic = self.cfg.SFDA.EMA_KEEP_RATE_dynamic
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint ).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluatorNEW(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluatorNEW(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_da_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.load_source_model(self.cfg.MODEL.LABLED_WEIGHTS)

        self.train_loop(self.start_iter, self.max_iter)

        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def load_source_model(self, path):
        DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(
            path, resume=False
        )
        DetectionCheckpointer(self.model_teacher_static, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(
            path, resume=False
        )
        DetectionCheckpointer(self.model_teacher_dynamic, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(
            path, resume=False
        )

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        with EventStorage(start_iter) as self.storage:
                for epoch in range(self.epochs):
                    if epoch >= self.cfg.SOLVER.WARM_UP_EPOCH and epoch % self.cfg.SOLVER.Period == 0:
                        logger.info(f"=====================================Swap Weight(iter:{self.iter})=================================")
                        self.swap_weight(self.model_teacher_static, self.model)
                        self.EMA_KEEP_RATE_dynamic = 0.999
                    try:
                        self.before_train()
                        for iter in range(start_iter, max_iter + 1):
                            self.iter = epoch * max_iter + iter + 1
                            self.before_step()
                            self.run_step_da()
                            self.after_step()
                    except Exception:
                        logger.exception("Exception during training:")
                        raise
                time_stamp = datetime.datetime.now()
                self.checkpointer.save(f"model_final_{time_stamp.strftime('%m-%d_%H-%M')}")
                self.after_train()
    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres # 还是通过物体概率score选取伪框

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def filter_pseudo_label(
            self, proposals_rpn_unsup_static, proposals_rpn_unsup_dynamic,
            iou_threshold=0.5
    ):
        list_instances = []
        for proposal_bbox_inst_1, proposal_bbox_inst_2 in zip(
                proposals_rpn_unsup_static, proposals_rpn_unsup_dynamic):

            boxes_list = []
            labels_list = []
            scores_list = []

            proposal_bbox_inst_boxes_1 = proposal_bbox_inst_1.gt_boxes.tensor
            proposal_bbox_inst_boxes_2 = proposal_bbox_inst_2.gt_boxes.tensor
            iou = box_iou(proposal_bbox_inst_boxes_1, proposal_bbox_inst_boxes_2)
            keeps = (iou > iou_threshold).nonzero()
            keeps_1, keeps_2 = keeps[:, 0], keeps[:, 1]

            image_size = proposal_bbox_inst_1._image_size
            gt_boxes1 = proposal_bbox_inst_1.gt_boxes.tensor[keeps_1, :] # h,w
            gt_boxes1[:, [0, 2]] /= image_size[1] #h, w
            gt_boxes1[:, [1, 3]] /= image_size[0]
            gt_boxes2 = proposal_bbox_inst_2.gt_boxes.tensor[keeps_2, :] # h,w
            gt_boxes2[:, [0, 2]] /= image_size[1] #h, w
            gt_boxes2[:, [1, 3]] /= image_size[0]

            boxes_list.append(gt_boxes1.tolist())
            boxes_list.append(gt_boxes2.tolist())
            labels_list.append(proposal_bbox_inst_1.gt_classes[keeps_1].tolist())
            labels_list.append(proposal_bbox_inst_2.gt_classes[keeps_2].tolist())
            scores_list.append(proposal_bbox_inst_1.scores[keeps_1].tolist())
            scores_list.append(proposal_bbox_inst_2.scores[keeps_2].tolist())

            # weights = [1, 1]
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None,
                                                            skip_box_thr=0.01, conf_type='max')
            boxes[:, [0, 2]] *= image_size[1]
            boxes[:, [1, 3]] *= image_size[0]

            proposal_bbox_inst_boxes = torch.as_tensor(boxes, dtype=torch.float32).cuda()
            proposal_bbox_inst_scores = torch.as_tensor(scores, dtype=torch.float32).cuda()
            proposal_bbox_inst_classes = torch.as_tensor(labels, dtype=torch.long).cuda()

            image_shape = proposal_bbox_inst_1.image_size
            new_proposal_inst = Instances(image_shape)
            # create box
            new_boxes = Boxes(proposal_bbox_inst_boxes)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst_classes
            new_proposal_inst.scores = proposal_bbox_inst_scores
            list_instances.append(new_proposal_inst)

        return list_instances

    def process_pseudo_label(
            self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================
    def run_step_da(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[DATeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)

        unlabel_data_q, unlabel_data_k, unlabel_data_weak = data
        unlabel_data_weak = self.remove_label(unlabel_data_weak)
        data_time = time.perf_counter() - start

        # remove unlabeled data labels
        unlabel_data_q = self.remove_label(unlabel_data_q) # strong 增强
        unlabel_data_k = self.remove_label(unlabel_data_k) # 弱增强


        if self.iter % self.cfg.SFDA.TEACHER_UPDATE_ITER == 0:
            self._update_teacher_model(mode_type="dynamic",
                                        keep_rate=self.EMA_KEEP_RATE_dynamic)

        record_dict = {}

        with torch.no_grad():
            (
                _,
                proposals_rpn_unsup_k_static,  # rpn output
                proposals_roih_unsup_k_static,  # roi_head output
                features_k_static
            ) = self.model_teacher_static(unlabel_data_k, branch="unsup_data_weak")

        with torch.no_grad():
            (
                _,
                proposals_rpn_unsup_k_dynamic,
                proposals_roih_unsup_k_dynamic,
                features_k_dynamic
            ) = self.model_teacher_dynamic(unlabel_data_k, branch="unsup_data_weak")


        cur_threshold = self.cfg.SFDA.BBOX_THRESHOLD

        joint_proposal_dict = {}

        # Pseudo_labeling for ROI head (bbox location/objectness)
        pesudo_proposals_roih_unsup_k_static, _ = self.process_pseudo_label(
            proposals_roih_unsup_k_static, cur_threshold, "roih", "thresholding"
        )
        pesudo_proposals_roih_unsup_k_dynamic, _ = self.process_pseudo_label(
            proposals_roih_unsup_k_dynamic, cur_threshold, "roih", "thresholding"
        )

        pesudo_proposals_roih_unsup_k = self.filter_pseudo_label(pesudo_proposals_roih_unsup_k_static,
                                                                 pesudo_proposals_roih_unsup_k_dynamic,
                                                                 iou_threshold=self.cfg.SFDA.PSEUDO_THRESHOLD)

        joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k  # 过滤后roi_head生成的预测框

        #  add pseudo-label to unlabeled data
        unlabel_data_q = self.add_label(
            unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
        )

        all_unlabel_data = unlabel_data_q

        record_all_unlabel_data, _, _, _ = self.model(
            all_unlabel_data, branch="supervised",
        )

        da_loss = {}

        for key in record_all_unlabel_data.keys():
            da_loss[key + "_pseudo"] = record_all_unlabel_data[
                key
            ]

        record_dict.update(da_loss)

        # weight losses
        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss":
                if key == "loss_rpn_loc_pseudo":
                    # pseudo bbox regression <- 0
                    loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SFDA.UNSUP_RPN_LOSS_WEIGHT
                    )
                elif key == "loss_box_reg_pseudo":
                    loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SFDA.UNSUP_ROI_LOSS_WEIGHT
                    )
                elif key[-6:] == "pseudo":  # unsupervised loss
                    loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SFDA.UNSUP_LOSS_WEIGHT
                    )
                else:  # supervised loss
                    loss_dict[key] = record_dict[key] * 1

        losses = sum(loss_dict.values())
        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                    for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, mode_type="static", keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        if mode_type == "dynamic":
            teacher_model = self.model_teacher_dynamic
        else:
            raise ValueError(f"{mode_type} is Unkown teacher model types.")
        for key, value in teacher_model.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        teacher_model.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher_static.load_state_dict(rename_model_dict)
            self.model_teacher_dynamic.load_state_dict(rename_model_dict)
        else:
            self.model_teacher_static.load_state_dict(self.model.state_dict())
            self.model_teacher_dynamic.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher_dynamic)
            return self._last_eval_results_teacher

        def test_and_save_results_teacher_help():
            self._last_eval_results_teacher_static = self.test(self.cfg, self.model_teacher_static)
            _last_eval_results_teacher_static = {
                k + "_teacher_static": self._last_eval_results_teacher_static[k]
                for k in self._last_eval_results_teacher_static.keys()
            }
            return _last_eval_results_teacher_static

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_teacher_help))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def swap_weight(self, modelS, modelT, with_guassin_noise=False):
        modelS_dict = modelS.state_dict()
        modelT_dict = modelT.state_dict()
        for key, value in modelS_dict.items():
            if key in modelT_dict.keys():
                modelS_dict[key].copy_(modelT_dict.get(key))
                modelT_dict[key].copy_(value)
            else:
                raise Exception("{} is not found in student model".format(key))

        modelS.load_state_dict(modelS_dict)
        modelT.load_state_dict(modelT_dict)






