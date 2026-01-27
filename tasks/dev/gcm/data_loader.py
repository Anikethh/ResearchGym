import numpy as np
import pandas as pd

class DataLoader(object):
    _info = {
        "adult": {"mono": {"education-num": 1,
                           "capital-gain": 1,
                           "age": 1,
                           "hours-per-week": 1},
                  "task": 1},
        "diabetes": {"mono": {"BMI": 1,
                              "HvyAlcoholConsump": 1,
                              "Smoker": 1,
                              "Age": 1},
                     "task": 1},
        "compas": {"mono": {"priors_count": 1,
                            "juv_fel_count": 1,
                            "juv_misd_count": 1,
                            "juv_other_count": 1},
                   "task": 1},
        "blog": {"mono": {
            "feature_50": 1,
                                "feature_51": 1,
                                "feature_52": 1,
                                "feature_53": 1,
                                "feature_55": 1,
                                "feature_56": 1,
                                "feature_57": 1,
                                "feature_58": 1
                          },
                 "task": 0},
        "auto": {"mono": {"Displacement": -1,
                                "Horsepower": -1,
                                "Weight": -1},
                 "task": 0},
        "heart": {"mono": {"trestbps": 1,
                           "chol": 1},
                  "task": 1},
        "loan": {"mono": {"feature_0": -1,
                          "feature_1": 1,
                          "feature_2": -1,
                          "feature_3": -1,
                          "feature_4": 1},
                 "task": 1}
    }

    def load_csv(self, name, mono_info):
        df = pd.read_csv(f'{name}')
        cols = df.columns
        rs, xs = [], []
        for n in cols:
            if n in mono_info:
                rs.append((mono_info[n] * df[n]).tolist())
            elif n != 'ground_truth':
                xs.append(df[n].tolist())
        r = np.float32(np.array([list(i) for i in zip(*rs)]))
        x = np.float32(np.array([list(i) for i in zip(*xs)]))
        y = np.float32(np.array(df['ground_truth'].tolist()))
        return x, r, y

    def load(self, name):
        mono_info = self._info[name]["mono"]
        task = self._info[name]["task"]
        train_x, train_r, train_y = self.load_csv(f'data/train_{name}.csv', mono_info)
        test_x, test_r, test_y = self.load_csv(f'data/test_{name}.csv', mono_info)
        x_dim = train_x.shape[1]
        r_dim = train_r.shape[1]
        if name == 'auto':
            train_y /= 10.0
            test_y /= 10.0
        if name == 'blog':
            train_x = np.log(1 + 20 * train_x)
            test_x = np.log(1 + 20 * test_x)
        return train_x, train_r, train_y, test_x, test_r, test_y, x_dim, r_dim, task
