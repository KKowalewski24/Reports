import pandas
import glob

MEASUREMENT_DATA_FILES = glob.glob("resources/measurement_data/*.xlsx")
LOCATION_DATA_FILES = glob.glob("resources/location_data/*.xlsx")
USE_COLS = [4, 5, 6, 7]


class Reader:

    def load_learning_data_series(self, normalize=True):
        # read
        series = [pandas.read_excel(data_file, usecols=USE_COLS)
                  for data_file in MEASUREMENT_DATA_FILES]
        # normalize and save to normal arrays
        data = []
        for df in series:
            if normalize:
                min_x = min([min(x['measurement x'] for i, x in df.iterrows()),
                             min(x['reference x'] for i, x in df.iterrows())])
                max_x = max([max(x['measurement x'] for i, x in df.iterrows()),
                             max(x['reference x'] for i, x in df.iterrows())])
                min_y = min([min(x['measurement y'] for i, x in df.iterrows()),
                             min(x['reference y'] for i, x in df.iterrows())])
                max_y = max([max(x['measurement y'] for i, x in df.iterrows()),
                             max(x['reference y'] for i, x in df.iterrows())])
            tmp = []
            for i, x in df.iterrows():
                if normalize:
                    tmp.append([(x['measurement x'] - min_x) / (max_x - min_x),
                                (x['measurement y'] - min_y) / (max_y - min_y),
                                (x['reference x'] - min_x) / (max_x - min_x),
                                (x['reference y'] - min_y) / (max_y - min_y)])
                else:
                    tmp.append([x['measurement x'], x['measurement y'], x['reference x'], x['reference y']])
            data.append(tmp)
        return data

    def load_testing_data(self, normalize=True):
        # it remembers maxs/mins from last normalization to 'denormalize' data later
        df = pandas.read_excel(LOCATION_DATA_FILES[0], usecols=USE_COLS)
        if normalize:
            self.min_x = min([min(x['measurement x'] for i, x in df.iterrows()),
                         min(x['reference x'] for i, x in df.iterrows())])
            self.max_x = max([max(x['measurement x'] for i, x in df.iterrows()),
                         max(x['reference x'] for i, x in df.iterrows())])
            self.min_y = min([min(x['measurement y'] for i, x in df.iterrows()),
                         min(x['reference y'] for i, x in df.iterrows())])
            self.max_y = max([max(x['measurement y'] for i, x in df.iterrows()),
                         max(x['reference y'] for i, x in df.iterrows())])
        tmp = []
        for i, x in df.iterrows():
            if i <= 1539:
                if normalize:
                    tmp.append([(x['measurement x'] - self.min_x) / (self.max_x - self.min_x),
                                (x['measurement y'] - self.min_y) / (self.max_y - self.min_y),
                                (x['reference x'] - self.min_x) / (self.max_x - self.min_x),
                                (x['reference y'] - self.min_y) / (self.max_y - self.min_y)])
                else:
                    tmp.append([x['measurement x'], x['measurement y'], x['reference x'], x['reference y']])
        return tmp

    def denormalize_testing_data(self, data):
        tmp = []
        for x in data:
            tmp.append([x[0] * (self.max_x - self.min_x) + self.min_x,
                        x[1] * (self.max_y - self.min_y) + self.min_y,
                        x[2] * (self.max_x - self.min_x) + self.min_x,
                        x[3] * (self.max_y - self.min_y) + self.min_y])
        return tmp;

    def denormalize_mlp_output(self, data):
        tmp = []
        for x in data:
            tmp.append([x[0] * (self.max_x - self.min_x) + self.min_x,
                        x[1] * (self.max_y - self.min_y) + self.min_y])
        return tmp;
