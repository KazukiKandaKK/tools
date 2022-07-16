'''main logic'''
from ml.execute_nn import Execute
import mlflow
mlflow.autolog()

if __name__ == '__main__':
    execute = Execute()
    execute.execute()
    execute.print_result()
