import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
from train_config import config as cfg
from sklearn.decomposition import PCA
#####prepare data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##prepare model

ROOT_PATH='../'

p_min,p_max=0,1


feature_file = '../lish-moa/train_features.csv'

features = pd.read_csv(feature_file)

test_file=os.path.join(ROOT_PATH,'lish-moa/test_features.csv')
test_features=pd.read_csv(test_file)

def preprocess(df):
    """Returns preprocessed data frame"""
    df = df.copy()
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})
    df = df.drop('sig_id', axis=1)
    return df

test_features=preprocess(test_features)

print(test_features.shape)

def predict_with_model(model,weights_list):


    cur_model_result=[]
    for weight in weights_list:
        print('predict with %s' % (weight))

        model.load_state_dict(torch.load(os.path.join('./models',weight), map_location=device))
        model.to(device)

        model.eval()


        with torch.no_grad():


            input_data=test_features.values
            input_data=torch.from_numpy(input_data).to(device).float()
            pre,_=model(input_data)

            pre=torch.sigmoid(pre)

            cur_model_result.append(pre.cpu().numpy())

    cur_model_result=np.stack(cur_model_result,axis=0)
    cur_model_result=np.mean(cur_model_result,axis=0)
    return cur_model_result


from lib.core.base_trainer.model import Complexer

model=Complexer()



weights_list=os.listdir('./models')


y_pred=predict_with_model(model,weights_list)




sub = pd.read_csv(os.path.join(ROOT_PATH,'./lish-moa/sample_submission.csv'))

sub.iloc[:,1:] = np.clip(y_pred,p_min,p_max)



# Save Submission
sub.to_csv('submission.csv', index=False)