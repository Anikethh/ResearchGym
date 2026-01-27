'''
2022.12.28 --zy 阳了在家写代码 
'''

import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadSuperPointsFromFile(object):
    """Load Points From File.

    Load superpoints points from file.

    Args:
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 multi_scale=1,
                 file_client_args=dict(backend='disk')):
      
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.multi_scale = multi_scale

    def _load_superpoints(self, spts_filename):
        """Private function to load superpoints data.

        Args:
            pts_filename (str): Filename of superpoints data.

        Returns:
            np.ndarray: An array containing superpoints data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            spts_bytes = self.file_client.get(spts_filename)
            superpoints = np.frombuffer(spts_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(spts_filename)
            superpoints = np.fromfile(spts_filename, dtype=np.long)
        return superpoints

    def __call__(self, results):
        """Call function to load superpoints data from file.

        Args:
            results (dict): Result dict containing superpointss data.

        Returns:
            dict: The result dict containing the superpoints data. \
                Added key and value are described below.
        """
        superpoints_filename = results['superpoints_filename']
        
        superpoints = self._load_superpoints(superpoints_filename)

        results['superpoints'] = superpoints
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ 
        return repr_str

