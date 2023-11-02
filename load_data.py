import pandas as pd
import json


class DataLoader:
    """
    Handle loading of data
    """

    def __init__(self, file_type="hour", randomState=7):
        """
        initialize Load_data
        """

        try:
            # Read the configuration file
            with open('config.json', 'r') as file:
                config = json.load(file)

            # Access configuration parameters
            self.hourly_data_path = config["hourly_data_path"]
            self.daily_data_path = config["daily_data_path"]
            self.train_split = config["train_split_ratio"]
            print(f'Data loaded successfully from config.json')
            if file_type == "hour":
                self.data = pd.read_csv(self.hourly_data_path)
            elif file == "day":
                self.data = pd.read_csv(self.daily_data_path)

            # Shuffle data
            self.data.sample(frac=1.0, random_state=randomState).reset_index(drop=True)
            print(f'CSV data loaded successfully.')
        except Exception as e:
            print(f'Error loading data: {e}')

    def fetchData(self):
        """
        Split data from a pandas dataframe.

        Returns:
            pd.DataFrame: dataframe of split.
        """
        try:
            split_index = int(len(self.data) * self.train_split)
            train = self.data[:split_index]
            test = self.data[split_index:]
            return self.data, train, test
        except Exception as e:
            print(f'Error loading data: {e}')
