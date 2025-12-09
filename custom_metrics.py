"""Custom evaluation metrics for PROVE unified training.

This module provides custom metrics that extend MMSegmentation's built-in metrics.
Currently includes:
- FWIoUMetric: Extends IoUMetric to add Frequency Weighted IoU (fwIoU)

Frequency Weighted IoU (fwIoU) Formula:
    fwIoU = sum(freq_i * IoU_i) where freq_i = area_label_i / total_area
    
This metric weights the IoU of each class by its frequency in the ground truth,
giving more importance to common classes and less to rare classes.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from mmseg.registry import METRICS
from mmseg.evaluation.metrics import IoUMetric


@METRICS.register_module()
class FWIoUMetric(IoUMetric):
    """IoU Metric with additional Frequency Weighted IoU (fwIoU) computation.
    
    This metric extends the standard IoUMetric to include fwIoU, which is
    computed as the sum of per-class IoU weighted by class frequency.
    
    Args:
        iou_metrics (list[str], optional): Metrics to compute. Defaults to ['mIoU'].
            Supported metrics: 'mIoU', 'mDice', 'mFscore', 'fwIoU'
        nan_to_num (int, optional): Replace NaN values. Defaults to None.
        beta (int, optional): Beta for F-score calculation. Defaults to 1.
        collect_device (str, optional): Device for collecting results. Defaults to 'cpu'.
        output_dir (str, optional): Output directory for results. Defaults to None.
        format_only (bool, optional): Only format results without computing metrics.
        prefix (str, optional): Prefix for metric names. Defaults to None.
    
    Example:
        >>> # In config file
        >>> val_evaluator = dict(
        ...     type='FWIoUMetric',
        ...     iou_metrics=['mIoU', 'fwIoU'],
        ...     prefix='val'
        ... )
    """
    
    # Extend allowed metrics to include fwIoU
    ALLOWED_METRICS = ['mIoU', 'mDice', 'mFscore', 'fwIoU']
    
    def __init__(self,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        # Validate metrics before passing to parent
        for metric in iou_metrics:
            if metric not in self.ALLOWED_METRICS:
                raise KeyError(f"metric {metric} is not supported. "
                              f"Supported metrics: {self.ALLOWED_METRICS}")
        
        # Separate fwIoU from metrics passed to parent
        self.compute_fwiou = 'fwIoU' in iou_metrics
        parent_metrics = [m for m in iou_metrics if m != 'fwIoU']
        
        # Ensure at least mIoU is computed if fwIoU is requested
        # (we need the IoU values to compute fwIoU)
        if self.compute_fwiou and 'mIoU' not in parent_metrics:
            parent_metrics.append('mIoU')
        
        # If only fwIoU was requested, ensure we still have a valid parent metric
        if not parent_metrics:
            parent_metrics = ['mIoU']
        
        # Store original metrics for output
        self._original_metrics = iou_metrics
        
        super().__init__(
            iou_metrics=parent_metrics,
            nan_to_num=nan_to_num,
            beta=beta,
            collect_device=collect_device,
            output_dir=output_dir,
            format_only=format_only,
            prefix=prefix,
            **kwargs
        )
    
    @staticmethod
    def compute_fwiou_from_areas(total_area_intersect: np.ndarray,
                                  total_area_union: np.ndarray,
                                  total_area_label: np.ndarray,
                                  nan_to_num: Optional[int] = None) -> float:
        """Compute Frequency Weighted IoU from area statistics.
        
        Args:
            total_area_intersect: Intersection area for each class.
            total_area_union: Union area for each class.
            total_area_label: Ground truth area for each class.
            nan_to_num: Value to replace NaN with.
            
        Returns:
            float: Frequency Weighted IoU value.
        """
        # Compute per-class IoU
        iou = total_area_intersect / total_area_union
        
        # Compute class frequencies (proportion of each class in ground truth)
        total_area = total_area_label.sum()
        if total_area == 0:
            return 0.0
        freq = total_area_label / total_area
        
        # Compute frequency weighted IoU
        # fwIoU = sum(freq_i * IoU_i) for all classes with valid IoU
        # Handle NaN values (classes with no predictions or ground truth)
        if nan_to_num is not None:
            iou = np.nan_to_num(iou, nan=nan_to_num)
        else:
            # For fwIoU, we typically ignore classes with NaN IoU
            valid_mask = ~np.isnan(iou)
            if not valid_mask.any():
                return 0.0
            # Renormalize frequencies for valid classes
            freq = freq * valid_mask
            if freq.sum() > 0:
                freq = freq / freq.sum()
            iou = np.nan_to_num(iou, nan=0.0)
        
        fwiou = (freq * iou).sum()
        return float(fwiou)
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute metrics including fwIoU.
        
        Args:
            results: List of results from process().
            
        Returns:
            Dict with metric names and values.
        """
        # Get parent metrics first
        metrics = super().compute_metrics(results)
        
        # Compute fwIoU if requested
        if self.compute_fwiou:
            # Aggregate results to get total areas
            total_area_intersect = np.zeros((self.num_classes,), dtype=np.float64)
            total_area_union = np.zeros((self.num_classes,), dtype=np.float64)
            total_area_label = np.zeros((self.num_classes,), dtype=np.float64)
            
            for result in results:
                total_area_intersect += result['area_intersect']
                total_area_union += result['area_union']
                total_area_label += result['area_label']
            
            # Compute fwIoU
            fwiou = self.compute_fwiou_from_areas(
                total_area_intersect,
                total_area_union,
                total_area_label,
                self.nan_to_num
            )
            
            metrics['fwIoU'] = fwiou * 100  # Convert to percentage like other metrics
        
        return metrics
