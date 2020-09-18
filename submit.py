import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from train_config import config as cfg
#####prepare data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ROOT='../stanford-covid-vaccine'
train = pd.read_json(ROOT+'/train.json', lines=True)
test = pd.read_json(ROOT+'/test.json', lines=True)
sample_sub = pd.read_csv(ROOT+'/sample_submission.csv')


#target columns
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):

    encode=np.array(df[cols]
                            .applymap(lambda seq: [token2int[x] for x in seq])
                            .values
                            .tolist()
        )


    return encode

public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

public_inputs=np.transpose(public_inputs,[0,2,1])
private_inputs=np.transpose(private_inputs,[0,2,1])


##prepare model

def predict_with_model(short_model,long_model,weights_list):


    cur_model_result=[]
    for weight in weights_list:
        print('predict with %s' % (weight))

        short_model.load_state_dict(torch.load(weight, map_location=device))
        short_model.to(device)
        long_model.load_state_dict(torch.load(weight, map_location=device))
        long_model.to(device)



        #### ingerence with batchsize 1 reduce mem problem
        res = []
        for k in tqdm(range(public_inputs.shape[0])):
            cur_pub_input = torch.from_numpy(public_inputs[k:k+1, ...]).to(device)
            cur_pub_res = short_model(None, cur_pub_input)
            res.append(cur_pub_res.data.cpu().numpy())
        short_model_preds =np.concatenate(res,axis=0)

        res = []
        for k in tqdm(range(private_inputs.shape[0])):
            cur_pub_input = torch.from_numpy(private_inputs[k:k + 1, ...]).to(device)
            cur_pub_res = long_model(None, cur_pub_input)
            res.append(cur_pub_res.data.cpu().numpy())
        long_model_preds = np.concatenate(res, axis=0)
        ###merge
        preds_gru=[]
        for df, preds in [(public_df, short_model_preds), (private_df, long_model_preds)]:
            for i, uid in enumerate(df.id):
                single_pred = preds[i]

                single_df = pd.DataFrame(single_pred, columns=target_cols)
                single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

                preds_gru.append(single_df)

        preds_gru_df = pd.concat(preds_gru)
        cur_model_result.append(preds_gru_df)


        print(preds_gru_df.shape)

    for i in range(1,len(weights_list)):

        cur_model_result[0]['reactivity']+=cur_model_result[i]['reactivity']
        cur_model_result[0]['deg_Mg_pH10'] += cur_model_result[i]['deg_Mg_pH10']
        cur_model_result[0]['deg_pH10'] += cur_model_result[i]['deg_pH10']
        cur_model_result[0]['deg_Mg_50C'] += cur_model_result[i]['deg_Mg_50C']
        cur_model_result[0]['deg_50C'] += cur_model_result[i]['deg_50C']



    folds=len(weights_list)
    blend_preds_df = pd.DataFrame()
    blend_preds_df['id_seqpos'] = cur_model_result[0]['id_seqpos']

    blend_preds_df['reactivity'] = cur_model_result[0]['reactivity']/folds
    blend_preds_df['deg_Mg_pH10'] = cur_model_result[0]['deg_Mg_pH10']/folds
    blend_preds_df['deg_pH10'] = cur_model_result[0]['deg_pH10']/folds
    blend_preds_df['deg_Mg_50C'] = cur_model_result[0]['deg_Mg_50C']/folds
    blend_preds_df['deg_50C'] = cur_model_result[0]['deg_50C']/folds

    return blend_preds_df


from lib.core.base_trainer.model import Complexer

short_model1=Complexer(pre_length=107)
long_model1=Complexer(pre_length=130)


weights_list=['./models/fold0_epoch_149_val_loss0.255521.pth',
              './models/fold1_epoch_149_val_loss0.264397.pth',
              './models/fold2_epoch_149_val_loss0.257018.pth',
              './models/fold3_epoch_149_val_loss0.257898.pth',
              './models/fold4_epoch_149_val_loss0.255823.pth']


res1=predict_with_model(short_model1,long_model1,weights_list)


submission = sample_sub[['id_seqpos']].merge(res1, on=['id_seqpos'])

submission.head()

# Saving the final output file
submission.to_csv(f"submission.csv", index=False)