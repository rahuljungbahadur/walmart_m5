import tensorflow as tf
import pandas as pd
import numpy as np

class ComputeSavePredictions:

    """
    This class has helper functions for computing and saving predictions given a model and a file name
    """

    def __init__(self, trained_model) -> None:
        
        ## Read in the dataset for generating predictions
        self.trained_model = trained_model

        self.input_ts_length = 1941

        self.cat_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        self.ts_columns = [f'd_{i}' for i in range(1,self.input_ts_length+1)]

        

        self.pred_tf_data = tf.data.experimental.make_csv_dataset(file_pattern='./complete_dataset_prediction.csv',
        batch_size=64, 
        column_names=[*self.cat_columns, *self.ts_columns],
        column_defaults=[*['']*6, *[0.]*self.input_ts_length],
        num_epochs=1,
        shuffle=False)

    def feature_creation_function(self):
        pass

    def compute_predictions(self):
        return self.trained_model.predict(self.pred_tf_data.map(lambda batch: self.feature_creation_function(batch=batch)))

    def get_validation_predictions(self):
        validation_set = pd.read_csv('./sales_train_validation.csv').iloc[:30490, -28:]
        sample_predictions_id = pd.read_csv('./sample_submission.csv').iloc[:30490, :1]
        combined_val_df = pd.concat([sample_predictions_id, validation_set], axis=1)
        combined_val_df.columns = ['id', *[f'F{i}' for i in range(1,29)]]
        return combined_val_df
    
    def get_evaluation_predictions(self):
        eval_set = pd.DataFrame(self.compute_predictions()[:, -1])
        sample_predictions_id = pd.read_csv('./sample_submission.csv').iloc[30490:, :1].reset_index()
        combined_eval_df = pd.concat([sample_predictions_id, eval_set], axis=0)
        combined_eval_df.columns = ['id', *[f'F{i}' for i in range(1,29)]]
        return combined_eval_df

    def get_complete_dataset(self):
        return pd.concat([self.get_validation_predictions(), self.get_evaluation_predictions()], axis=0)





