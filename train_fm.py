#%%
import xlearn as xl


#%% Training task
fm_model = xl.create_fm()  # create model
fm_model.setTrain("./data/train_fold1.csv")
fm_model.setValidate("./data/test_fold1.csv")
fm_model.disableNorm()
# fm_model.disableEarlyStop()  # enable early stop
# fm_model.setOnDisk()  # enable batch loading from disk

param = {'task':'reg',
         'lr':0.1,
         'lambda':0.01,
         'epoch':100,
         'opt':'adagrad',
         'k':10,
         'stop_window':3,
         'nthread':4,
         'metric': 'rmse'
         }

        #  'block_size':512,

#%% Train model
fm_model.fit(param, "./model/fm02.out")
