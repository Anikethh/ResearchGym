# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet3d.datasets.builder import PIPELINES




@PIPELINES.register_module()
class SPPointSample(object):
    """Point sample.

    Sampling data to a certain number.

    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool, optional): Whether the sampling is with or without
            replacement. Defaults to False.
    """

    def __init__(self, num_points, based_on_superpoints=False, sample_range=None, replace=False):
        self.num_points = num_points
        self.sample_range = sample_range
        self.based_on_superpoints = based_on_superpoints
        self.replace = replace

    def _points_random_sampling(self,
                                points,
                                num_samples,
                                sample_range=None,
                                replace=False,
                                return_choices=False):
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (np.ndarray | :obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool, optional): Sampling with or without replacement.
                Defaults to None.
            return_choices (bool, optional): Whether return choice.
                Defaults to False.
        Returns:
            tuple[np.ndarray] | np.ndarray:
                - points (np.ndarray | :obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if type(self.num_points) is float:
            num_samples = int(np.random.uniform(self.num_points, 1.) * points.shape[0])

        if not replace:
            replace = (points.shape[0] < num_samples)
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            dist = np.linalg.norm(points.tensor, axis=1)
            far_inds = np.where(dist >= sample_range)[0]
            near_inds = np.where(dist < sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(
                    far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, results):
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        
        if self.based_on_superpoints:
            points = results['points']
            superpoints = results['superpoints']
            indices = np.arange(points.shape[0])
            superpoints_ids = np.unique(superpoints)
            choices = []
            for sp_ids in superpoints_ids:
                sps_mask = superpoints == sp_ids
                pts_in_spts = points[sps_mask]
                inds_in_spts = indices[sps_mask]
                _, cur_inds_in_spts_select = self._points_random_sampling(pts_in_spts,
                                                            self.num_points, return_choices=True)
                cur_inds_in_spts = inds_in_spts[cur_inds_in_spts_select]
                cur_inds_in_spts = inds_in_spts[cur_inds_in_spts_select]
                choices.append(cur_inds_in_spts)
            choices = np.concatenate(choices,axis=0)    
            results['points'] = points[choices]

            pts_instance_mask = results.get('pts_instance_mask', None)
            pts_semantic_mask = results.get('pts_semantic_mask', None)
            pts_superpoints = results.get('superpoints', None)


            if pts_instance_mask is not None:
                pts_instance_mask = pts_instance_mask[choices]
                results['pts_instance_mask'] = pts_instance_mask

            if pts_semantic_mask is not None:
                pts_semantic_mask = pts_semantic_mask[choices]
                results['pts_semantic_mask'] = pts_semantic_mask

            if pts_superpoints is not None:
                pts_superpoints = pts_superpoints[choices]
                superpoints_ids, pts_superpoints = np.unique(pts_superpoints, return_inverse=True)
                results['superpoints'] = pts_superpoints
        else:
            points = results['points']
            points, choices = self._points_random_sampling(
                points,
                self.num_points,
                self.sample_range,
                self.replace,
                return_choices=True)
            results['points'] = points

            pts_instance_mask = results.get('pts_instance_mask', None)
            pts_semantic_mask = results.get('pts_semantic_mask', None)
            pts_superpoints = results.get('superpoints', None)


            if pts_instance_mask is not None:
                pts_instance_mask = pts_instance_mask[choices]
                results['pts_instance_mask'] = pts_instance_mask

            if pts_semantic_mask is not None:
                pts_semantic_mask = pts_semantic_mask[choices]
                results['pts_semantic_mask'] = pts_semantic_mask

            if pts_superpoints is not None:
                pts_superpoints = pts_superpoints[choices]
                superpoints_ids, pts_superpoints = np.unique(pts_superpoints, return_inverse=True)
                results['superpoints'] = pts_superpoints
            
        

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points},'
        repr_str += f' sample_range={self.sample_range},'
        repr_str += f' replace={self.replace})'

        return repr_str




