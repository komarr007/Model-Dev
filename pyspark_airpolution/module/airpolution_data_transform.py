
import airpolution_data_constant
from pyspark.sql.functions import concat, date_format, col, to_timestamp, lit, to_date, when, mean, coalesce, avg, log
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
import numpy as np

_FEATURE_KEY = airpolution_data_constant.FEATURE_KEY
_LABEL_KEY = airpolution_data_constant.LABEL_KEY
_ALL_KEY = airpolution_data_constant.ALL_KEY

_transformed_name = airpolution_data_constant.transformed_name
_vectorize_name = airpolution_data_constant.vectorize_name


def change_time_format(inputs):
  """This Function Changing the time format"""

  df = inputs.withColumn('Time', date_format('Time', 'HH:mm:ss'))

  return df

def merge_date_time(inputs):
  """This Function Merging Date & Time Column"""

  datetime_col = concat(inputs.Date, lit(" "), inputs.Time)
  df = inputs.withColumn("datetime", to_timestamp(datetime_col, "dd/MM/yyyy HH:mm:ss"))

  return df

def clean_date_format(inputs):
  """This Function Fix Date & Time Type"""

  df = inputs.withColumn("Date", to_date("Date", "dd/MM/yyyy"))

  return df

def clean_outlier(inputs):
  """This Function clean outlier and fill it with mean"""

  df = inputs.replace(-200, None)

  for i in df.columns:
    if i not in ["Date","Time","datetime"]:
 
      mean_col_value = df.select(mean(col(i))).collect()[0][0]

      df = df.na.fill(mean_col_value, i)

  return df

def drop_duplicate(inputs):
  """This Function deleting duplicate value in dataframe"""

  df = inputs.dropDuplicates()

  return df

def data_distribution(inputs):
  """This function transforms the data distribution"""

  column_transform_type = {
      'CO(GT)': 'Log',
      'PT08_S1(CO)': 'Reciprocal',
      'NMHC(GT)': 'Log',
      'C6H6(GT)': 'Log',
      'PT08_S2(NMHC)': 'Log',
      'NOx(GT)': 'Log',
      'PT08_S3(NOx)': 'Log',
      'NO2(GT)': 'Original',
      'PT08_S4(NO2)': 'Original',
      'PT08_S5(O3)': 'Log',
      'T': 'Original',
      'RH': 'Original',
      'AH': 'Original'
  }

  transformed_df = inputs

  for col_name, transform_type in column_transform_type.items():
      if transform_type == "Log":
          transformed_df = transformed_df.withColumn(col_name, log(col(col_name)))
      elif transform_type == "Reciprocal":
          transformed_df = transformed_df.withColumn(col_name, 1 / col(col_name))

  return transformed_df

def normalize_data(inputs):
    """Normalize the data"""

    column_to_remove = ['Date', 'Time', 'datetime']
    column_to_use = [i for i in inputs.columns if i not in column_to_remove]

    df = inputs

    for column in column_to_use:
        assembler = VectorAssembler(inputCols=[column], outputCol=_vectorize_name(column))
        minmax = MinMaxScaler(inputCol=_vectorize_name(column), outputCol=_transformed_name(column))
        pipeline = Pipeline(stages=[assembler, minmax])
        df = pipeline.fit(df).transform(df).drop(_vectorize_name(column))

    return df

def preprocessing_fn(inputs, dist_transform=False, normalize=False):
  """Main Function"""
  
  df = change_time_format(inputs)
  df = merge_date_time(df)
  df = clean_date_format(df)
  df = clean_outlier(df)
  df = drop_duplicate(df)

  if dist_transform:
    df = data_distribution(df)

  if normalize:
    df = normalize_data(df)

  return df
