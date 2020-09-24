import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
from train_config import config as cfg
#####prepare data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TTA=True
ROOT='../stanford-covid-vaccine'
train = pd.read_json(ROOT+'/train.json', lines=True)
test = pd.read_json(ROOT+'/test.json', lines=True)
sample_sub = pd.read_csv(ROOT+'/sample_submission.csv')

aug_df=pd.read_csv('../stanford-covid-vaccine/aug_data.csv')

#target columns
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}


def parse_file( train):
    # target columns

    token2int = {x: i for i, x in enumerate('().ACGUBEHIMSX')}

    def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
        encode = np.array(df[cols]
                          .applymap(lambda seq: [token2int[x] for x in seq])
                          .values
                          .tolist()
                          )

        return encode


    train_inputs = preprocess_inputs(train)

    reconstructed_data=[]
    for index in range(len(train)):
        iid = train.iloc[index]['id']


        bpp_path = os.path.join('../stanford-covid-vaccine/bpps', iid + '.npy')

        image = np.load(bpp_path)

        data = train_inputs[index]


        data = np.transpose(data, [1, 0])  ##shape [n,107,3)


        bpp_max = np.expand_dims(np.max(image, axis=-1), -1)
        bpp_sum = np.expand_dims(np.sum(image, axis=-1), -1)

        bpps_nb_mean = 0.077522  # mean of bpps_nb across all training data
        bpps_nb_std = 0.08914  # std of bpps_nb across all training data
        bpps_nb = (image > 0).sum(axis=0) / image.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std

        bpps_nb = np.expand_dims(bpps_nb, axis=-1)
        data = np.concatenate([data, bpp_max, bpp_sum, bpps_nb], axis=1)

        reconstructed_data.append(data)

    return  np.array(reconstructed_data,dtype=np.float32)


def get_sample(df,index):

    cur_df=df.iloc[index:index+1]
    if TTA:
        target_df = cur_df.copy()
        new_df = aug_df[aug_df['id'].isin(target_df['id'])]

        del target_df['structure']
        del target_df['predicted_loop_type']
        new_df = new_df.merge(target_df, on=['id', 'sequence'], how='left')

        cur_df['cnt'] = df['id'].map(new_df[['id', 'cnt']].set_index('id').to_dict()['cnt'])
        cur_df['log_gamma'] = 100
        cur_df['score'] = 1.0
        cur_df = cur_df.append(new_df[cur_df.columns])

    data=parse_file(cur_df)

    return np.array(data,dtype=np.float32)

public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()


public_inputs = parse_file(public_df)
private_inputs = parse_file(private_df)


def aug_data( df, aug_df):
    target_df = df.copy()
    new_df = aug_df[aug_df['id'].isin(target_df['id'])]

    del target_df['structure']
    del target_df['predicted_loop_type']
    new_df = new_df.merge(target_df, on=['id', 'sequence'], how='left')

    df['cnt'] = df['id'].map(new_df[['id', 'cnt']].set_index('id').to_dict()['cnt'])
    df['log_gamma'] = 100
    df['score'] = 1.0
    df = df.append(new_df[df.columns])
    return df

##prepare model

def predict_with_model(short_model,long_model,weights_list):

    cur_model_result=[]
    for weight in weights_list:
        print('predict with %s' % (weight))

        short_model.load_state_dict(torch.load(weight, map_location=device))
        short_model.to(device)
        short_model.eval()
        long_model.load_state_dict(torch.load(weight, map_location=device))
        long_model.to(device)
        long_model.eval()

        #### ingerence with batchsize 1 reduce mem problem
        res = []
        for k in tqdm(range(public_inputs.shape[0])):

            cur_input=get_sample(public_df,k)

            cur_pub_input = torch.from_numpy(cur_input).to(device)
            cur_pub_res = short_model( cur_pub_input)
            cur_pub_res = torch.mean(cur_pub_res,dim=0,keepdim=True)
            res.append(cur_pub_res.data.cpu().numpy())
        short_model_preds =np.concatenate(res,axis=0)

        res = []
        for k in tqdm(range(private_inputs.shape[0])):
            cur_input=get_sample(private_df,k)

            cur_pub_input = torch.from_numpy(cur_input).to(device)
            cur_pub_res = long_model( cur_pub_input)
            cur_pub_res = torch.mean(cur_pub_res, dim=0, keepdim=True)
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



models=[{'model_name':'gru',
         'model':Complexer,
         'mtype':0,
         "weights":['./models/gru_fold0_epoch_36_val_loss0.227445.pth',
                    './models/gru_fold1_epoch_34_val_loss0.230443.pth',
                    './models/gru_fold2_epoch_36_val_loss0.234735.pth',
                    './models/gru_fold3_epoch_30_val_loss0.233050.pth',
                    './models/gru_fold4_epoch_33_val_loss0.232526.pth',
                    './models/gru_fold5_epoch_32_val_loss0.229593.pth',
                    './models/gru_fold6_epoch_35_val_loss0.238407.pth',
                    './models/gru_fold7_epoch_31_val_loss0.223315.pth',
                    './models/gru_fold8_epoch_35_val_loss0.224473.pth',
                    './models/gru_fold9_epoch_38_val_loss0.234292.pth']},


            ]

for model in models:

    model_function=model['model']

    short_model1=model_function(pre_length=107,mtype=model['mtype'])
    long_model1=model_function(pre_length=130,mtype=model['mtype'])


    weights_list=model['weights']


    res1=predict_with_model(short_model1,long_model1,weights_list)


    submission = sample_sub[['id_seqpos']].merge(res1, on=['id_seqpos'])

    submission.head()
    # Saving the final output file
    submission.to_csv("%s_submission.csv"%(model['model_name']), index=False)