import argparse
import pandas as pd
import xgboost as regressor

def predict_ghi(config):
    file_name='xgboost_model_lux_series.json'
    new_xgb=regressor.XGBRegressor()
    new_xgb.load_model('xgboost_model_lux_series.json')
    print(f'  Detials of the loaded model \n  {new_xgb}\n\n')
    if config.flag:
        csv_file = pd.read_csv(config.csv,sep=';')
        if len(csv_file.columns)==9:
            lux_values = csv_file.iloc[:,:].values
            #timestamp = csv_file.iloc[:;0].values #append timestamp
            predictions = new_xgb.predict(lux_values)
            print(f'  Correct input, wait untill the model predicts the ghi values and save them to csv...')
            output_file=pd.DataFrame(columns=["Predictions"])
            output_file["Predictions"]=predictions
            output_file.to_csv('Predicted_ghi.csv', index=False)
            print(f"  Writing to file done, file is saved as Predicted_ghi.csv\nOutput GHI Values:\n")
            # print(predictions)
            return predictions
        else:
            print(f'  EError!\n Please Provide a csv file with lux values with command python3 predict_ghi.py --csv=\"file_name\" --flag=True \nMake sure your csv file has 9 columns with lux values\n  there are {len(csv_file.columns)} Cols in the file.')
            return "  Error!!! Retry with required inputs"

    else:
        lux_list = [eval(config.values)]
        print(f'  list is given and the values are: {lux_list}\n')
        if len(eval(config.values))!=9:
            print(f'  EError!\n  Please Enter a list of 9 lux values like this python3 predict_ghi.py --values="[1330,1408,3583,1262,400,1278,2427,1531,1366]"\n\n  or if you have csv file then\n  python3 predict_ghi.py --csv=\"file_name\" --flag=True\n')
            return "  Error!!! Retry with required inputs"
        else:
            predictions = new_xgb.predict(lux_list)
            print(f'  Correct input, wait untill the model predicts the ghi values and save them to csv...')
            output_file=pd.DataFrame(columns=["Predictions"])
            output_file["Predictions"]=predictions
            output_file.to_csv('Predicted_ghi.csv', index=False)
            print(f"  Writing to file done, file is saved as 'Predicted_ghi.csv'\n  Output GHI Values:\n")
            # print(predictions)
            return predictions




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take a csv file having 8 columns with values for lux or a list of 8 values and predict the ghi value(s).")
    parser.add_argument('--csv', type=str)
    parser.add_argument('--flag', type=bool, default=False)
    parser.add_argument('--values', type=str, default='[]')
    config= parser.parse_args()
    print(predict_ghi(config))
