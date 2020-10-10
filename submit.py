import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
from train_config import config as cfg
#####prepare data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##prepare model

ROOT_PATH='../'

p_min,p_max=0,1

test_file=os.path.join(ROOT_PATH,'lish-moa/test_features.csv')
test_features=pd.read_csv(test_file)



dose = np.array(test_features['cp_dose'].values == 'D1', dtype=np.float32)

test_features['cp_dose_encoded'] = dose

test_features_input = test_features.drop(['sig_id','cp_dose','cp_type'],axis=1)

print(test_features.shape)

def predict_with_model(model,weights_list):


    cur_model_result=[]
    for weight in weights_list:
        print('predict with %s' % (weight))

        model.load_state_dict(torch.load(weight, map_location=device))
        model.to(device)




        with torch.no_grad():


            input_data=test_features_input.values
            input_data=torch.from_numpy(input_data).to(device).float()
            pre=model(input_data)

            pre=torch.sigmoid(pre)

            cur_model_result.append(pre.cpu().numpy())

    cur_model_result=np.stack(cur_model_result,axis=0)
    cur_model_result=np.mean(cur_model_result,axis=0)
    return cur_model_result


from lib.core.base_trainer.model import Complexer

model=Complexer()



weights_list=['./models/fold0_epoch_46_val_loss0.016625.pth']


y_pred=predict_with_model(model,weights_list)




sub = pd.read_csv(os.path.join(ROOT_PATH,'./lish-moa/sample_submission.csv'))

sub.iloc[:,1:] = np.clip(y_pred,p_min,p_max)

# Set ctl_vehicle to 0
sub[test_features['cp_type'] == 'ctl_vehicle'][1:] = 0

# Save Submission
sub.to_csv('submission.csv', index=False)