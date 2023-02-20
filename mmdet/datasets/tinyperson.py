import contextlib
import io
import itertools
import logging
import shutil
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from .builder import DATASETS
from .coco import CocoDataset
from ..core.evaluation.tiny.split_and_merge import merge_det_result
from ..core.evaluation.tiny.tinyperson_eval import COCOeval, COCOeval_MR


@DATASETS.register_module()
class TinyPersonDataset(CocoDataset):
    def __init__(self,
                 ann_file,
                 merge_config=None,
                 **kwargs):

        self.train_ignore_as_bg = True
        self.merge_config = merge_config
        super().__init__(ann_file, **kwargs)

    # CHECKED
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            # NEW
            if self.train_ignore_as_bg and ann.get('ignore', False):
                continue
            # END NEW
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 eval_mr=True
                 # use_location_metric=False,
                 # location_kwargs={}
                 ):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        tiny_result_str = ''  # NEW

        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            # TODO: check this
            try:
                # NEW
                shutil.copy(result_files[metric], './pred.json')
                if self.merge_config is not None:
                    cocoGt, result_files[metric] = merge_det_result(
                        result_files[metric], self.ann_file,
                        self.merge_config.get("origin_ann_file", None), self.merge_config.get("nms_thr", 0.5)
                    )
                    shutil.copy(result_files[metric], './merge.json')
                predictions = mmcv.load(result_files[metric])
                # END NEW

                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log('The testing results of the whole dataset is empty.', logger=logger, level=logging.ERROR)
                break

            # TODO L: use this?
            # # NEW
            # if use_location_metric:
            #     location_eval = LocationEvaluator(**location_kwargs)
            #     print(location_eval.__dict__)
            #     LocationEvaluator.add_center_from_bbox_if_no_point(cocoDt)
            #     res_set = location_eval(cocoDt, cocoGt)
            #     location_eval.summarize(res_set, cocoGt, print_func=partial(print_log, logger=logger))
            #     continue
            # END NEW

            iou_type = 'bbox' if metric == 'proposal' else metric
            # cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = np.array(iou_thrs)
            print({k: v for k, v in cocoEval.params.__dict__.items() if k not in ['imgIds']})

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }

            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                # if metric_items is None:
                #     metric_items = [
                #         'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                #     ]
                # for metric_item in metric_items:
                #     key = f'{metric}_{metric_item}'
                #     val = float(
                #         f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                #     )
                #     eval_results[key] = val
                # ap = cocoEval.stats[:6]
                # eval_results[f'{metric}_mAP_copypaste'] = (
                #     f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                #     f'{ap[4]:.3f} {ap[5]:.3f}')

                # NEW
                ap = cocoEval.stats
                tiny_result_str += (
                    f'{100 * ap[8]:.3f} {100 * ap[9]:.3f} {100 * ap[10]:.3f} {100 * ap[11]:.3f} '
                    f'{100 * ap[12]:.3f} {100 * ap[1]:.3f} {100 * ap[15]:.3f} {100 * ap[7]:.3f} '
                )  # AP: tiny_50 tiny1_50 tiny2_50 tiny3_50 small_50 tiny_25 tiny_75 all_50

                tiny_metric_names = {
                    'AP_tiny_50': 8,
                    'AP_tiny1_50': 9,
                    'AP_tiny2_50': 10,
                    'AP_tiny3_50': 11,
                    'AP_small_50': 12,
                    'AP_tiny_25': 1,
                    'AP_tiny_75': 15,
                    'AP_50': 7,

                    # 'AR_tiny_50': 29,
                    # 'AR_tiny1_50': 30,
                    # 'AR_tiny2_50': 31,
                    # 'AR_tiny3_50': 32,
                    # 'AR_small_50': 33,
                    # 'AR_tiny_25': 28,
                    # 'AR_tiny_75': 42,
                    # 'AR_50': 34,
                }
                for key, index in tiny_metric_names.items():
                    eval_results[key] = ap[index]
                # END NEW

            # NEW
            if eval_mr:
                cocoEval_MR = COCOeval_MR(cocoGt, cocoDt, standard='tiny', filter_type='size')

                mr = []
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    for id in range(cocoEval_MR.num_scales):
                        cocoEval_MR.evaluate(id)
                        cocoEval_MR.accumulate()
                        res = cocoEval_MR.summarize(id)
                        mr.extend(res)
                print_log('\n' + redirect_string.getvalue(), logger=logger)
                tiny_result_str += (
                    f'{100 * mr[10]:.3f} {100 * mr[1]:.3f} {100 * mr[4]:.3f} {100 * mr[7]:.3f} '
                    f'{100 * mr[13]:.3f} {100 * mr[9]:.3f} {100 * mr[11]:.3f} {100 * mr[16]:.3f} '
                )  # MR: tiny_50 tiny1_50 tiny2_50 tiny3_50 small_50 tiny_25 tiny_75 all_50
            # END NEW

        # NEW
        try:
            tiny_result_str += (
                f'{100 * ap[29]:.3f} {100 * ap[30]:.3f} {100 * ap[31]:.3f} {100 * ap[32]:.3f} '
                f'{100 * ap[33]:.3f} {100 * ap[22]:.3f} {100 * ap[36]:.3f} {100 * ap[28]:.3f} '
            )  # AR: tiny_50 tiny1_50 tiny2_50 tiny3_50 small_50 tiny_25 tiny_75 all_50
        except Exception as e:
            print(e)
        eval_results['tiny_result'] = tiny_result_str
        # END NEW

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results
