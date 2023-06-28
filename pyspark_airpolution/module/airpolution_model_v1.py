
import airpolution_data_constant
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

_FEATURE_KEY = airpolution_data_constant.FEATURE_KEY
_LABEL_KEY = airpolution_data_constant.LABEL_KEY
_ALL_KEY = airpolution_data_constant.ALL_KEY
FEATURE_KEY_TN = [
    'CO(GT)_tn',
    'PT08_S1(CO)_tn',
    'NMHC(GT)_tn',
    'C6H6(GT)_tn',
    'PT08_S2(NMHC)_tn',
    'NOx(GT)_tn',
    'PT08_S3(NOx)_tn',
    'NO2(GT)_tn',
    'PT08_S4(NO2)_tn',
    'PT08_S5(O3)_tn',
    'RH_tn',
    'AH_tn']


def transform_data(input, assembler):
    """Vectorizing the data"""

    print("Vectorizing the data")
    
    transformed = assembler.transform(input)
    return transformed

def splitting_data(input):
    """Splitting the data"""

    print("Splitting the data")
    (train_data, test_data) = input.randomSplit([0.8, 0.2])

    return (train_data, test_data)

def train_model(model, input):
    """Inputing model from user"""

    trained_model = model.fit(input)
    return trained_model

def eval_model(pred_model):
    """evaluate model"""

    rmse = RegressionEvaluator(labelCol=_LABEL_KEY, predictionCol="prediction", metricName="rmse")
    rmse = rmse.evaluate(pred_model)
    mae = RegressionEvaluator(labelCol=_LABEL_KEY, predictionCol="prediction", metricName="mae")
    mae = mae.evaluate(pred_model)
    r2 = RegressionEvaluator(labelCol=_LABEL_KEY, predictionCol="prediction", metricName="r2")
    r2 = r2.evaluate(pred_model)

    print("RMSE: ", rmse)
    print("MAE: ", mae)
    print("R-squared: ", r2)
    
def model_fn(model_list, input, assembler_transform):

    if assembler_transform:
            assembler = VectorAssembler(inputCols=FEATURE_KEY_TN, outputCol="features")
    else:
            assembler = VectorAssembler(inputCols=_FEATURE_KEY, outputCol="features")

    transformed_data = transform_data(input, assembler)
    train_data, test_data = splitting_data(transformed_data)
    
    for model in model_list:
        print("\n,", model)
        trained_model = train_model(model, train_data)
        pred_model = trained_model.transform(test_data)
        model_evaluation = eval_model(pred_model)
        print("\n")

    return trained_model
