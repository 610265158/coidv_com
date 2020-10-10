import sklearn.model_selection
import pandas as pd



feature_file='../lish-moa/train_features.csv'
target_file='../lish-moa/train_targets_scored.csv'
noscore_target='../lish-moa/train_targets_nonscored.csv'



train_feature=pd.read_csv(feature_file)
target=pd.read_csv(target_file)
extra_target=pd.read_csv(noscore_target)


train_feature['fold']=-1
kf=sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

for fold,(train_index , test_index) in enumerate(kf.split(train_feature)):
    train_feature['fold'][test_index]=fold

train_feature.to_csv('folded.csv',index=False)