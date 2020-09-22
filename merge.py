import pandas as pd


df_1=pd.read_csv('./gru_submission.csv')

df_2=pd.read_csv('./lstm_submission.csv')






model_result=[df_1,df_2]
for i in range(1, len(model_result)):
    model_result[0]['reactivity'] += model_result[i]['reactivity']
    model_result[0]['deg_Mg_pH10'] += model_result[i]['deg_Mg_pH10']
    model_result[0]['deg_pH10'] += model_result[i]['deg_pH10']
    model_result[0]['deg_Mg_50C'] += model_result[i]['deg_Mg_50C']
    model_result[0]['deg_50C'] += model_result[i]['deg_50C']

folds = len(model_result)
blend_preds_df = pd.DataFrame()
blend_preds_df['id_seqpos'] = model_result[0]['id_seqpos']

blend_preds_df['reactivity'] = model_result[0]['reactivity'] / folds
blend_preds_df['deg_Mg_pH10'] = model_result[0]['deg_Mg_pH10'] / folds
blend_preds_df['deg_pH10'] = model_result[0]['deg_pH10'] / folds
blend_preds_df['deg_Mg_50C'] = model_result[0]['deg_Mg_50C'] / folds
blend_preds_df['deg_50C'] = model_result[0]['deg_50C'] / folds


blend_preds_df.to_csv("submission.csv", index=False)