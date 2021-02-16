#%%
import xlearn as xl

#%% Predict
fm_model = xl.create_fm()
fm_model.setTest("./data/test_fold1.csv")
fm_model.predict("./model/fm01.out", "./output/output1.txt")