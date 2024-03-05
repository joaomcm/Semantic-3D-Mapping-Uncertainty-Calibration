import pandas as pd
import numpy as np
import json


SEED = 0


def get_larger_test_and_validation_scenes():
    
    df = pd.read_csv('./scenes_and_contents_20_classes.csv')
    np.random.seed(0)
    series = df.scene
    scenes = sorted(np.random.choice(sorted(series.to_numpy().tolist()),size = 100,replace = False))    
    testable_scenes = series[~series.str.split('_',expand = True).iloc[:,0].isin(pd.Series(scenes).str.split('_',expand = True).iloc[:,0])]
    new_test_scenes = sorted(np.random.choice(sorted(testable_scenes.to_numpy().tolist()),size = 100,replace = False))

    return scenes,new_test_scenes

def get_smaller_balanced_validation_scenes():
    return sorted(['scene0412_01','scene0025_02','scene0700_01','scene0046_01','scene0203_00','scene0697_01',
     'scene0462_00', 'scene0406_00', 'scene0378_00','scene0278_01'])


def get_original_small_validation_scenes():
    return sorted(['scene0063_00', 'scene0144_00', 'scene0203_02', 'scene0314_00', 'scene0356_01',
                      'scene0414_00', 'scene0474_00', 'scene0578_01', 'scene0629_01', 'scene0700_02'])

def get_smaller_test_scenes():
    return sorted(['scene0518_00', 'scene0146_01', 'scene0355_01', 'scene0568_02',
       'scene0651_02', 'scene0030_01', 'scene0593_01',
       'scene0685_00', 'scene0645_00'])

def get_learned_calibration_validation_scenes():
    return sorted(['scene0025_00', 'scene0648_00', 'scene0256_00', 'scene0207_00',
       'scene0203_01', 'scene0406_01', 'scene0651_00', 'scene0355_00',
       'scene0334_01', 'scene0697_00'])

def get_COLORS():
    COLORS = np.array([
    [0,0,0],[151,226,173],[174,198,232],[31,120,180],[255,188,120],[188,189,35],[140,86,74],[255,152,151],[213,39,40],[196,176,213],[148,103,188],[196,156,148],
    [23,190,208],[247,183,210],[218,219,141],[254,127,14],[227,119,194],[158,218,229],[43,160,45],[112,128,144],[82,83,163]
    ]).astype(np.uint8)
    return COLORS

def get_classes():
    classes = ['irrelevant','wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture',
           'counter','desk','curtain','refrigerator','shower curtain','toilet','sink','bathtub','otherfurniture']
    return classes


def derive_learned_splits():
    df = pd.read_csv('./scenes_and_contents_20_classes.csv', index_col = 0)
    np.random.seed(0)
    series = df.scene
    scenes = sorted(np.random.choice(sorted(series.to_numpy().tolist()),size = 100,replace = False))    


    refined_scenes = pd.Series(scenes).str.split('_',expand=  True)
    refined_scenes.loc[:,'scan'] = scenes
    refined_scenes.columns = ['room','scan_num','scan']

    unique_scenes = refined_scenes.iloc[:,0].unique()
    final = False
    while (not final):
        min_val = 2
        val_scenes = sorted(np.random.choice(unique_scenes,len(unique_scenes)//5,replace = False))

        val_scans = refined_scenes.loc[refined_scenes.room.isin(val_scenes),'scan'].tolist()
        train_scans = refined_scenes.loc[~refined_scenes.room.isin(val_scenes),'scan'].tolist()

        val_sum = (df.loc[df.scene.isin(val_scans),df.columns[:21]].sum(axis = 0)>=min_val).sum()

        train_sum = (df.loc[df.scene.isin(train_scans),df.columns[:21]].sum(axis = 0)>=min_val).sum()
        print(val_sum,train_sum)
        final = (val_sum ==21) and (train_sum == 21)
    return val_scans,train_scans

def get_fixed_train_and_val_splits():
    train_scans = ['scene0030_00','scene0050_01', 'scene0063_00', 'scene0077_00', 'scene0077_01', 'scene0081_00',
                    'scene0084_01', 'scene0084_02', 'scene0086_01', 'scene0088_02', 'scene0088_03', 'scene0100_00',
                    'scene0144_01', 'scene0146_00', 'scene0164_00', 'scene0164_03', 'scene0169_00', 'scene0187_01',
                    'scene0193_00', 'scene0203_00', 'scene0203_01', 'scene0203_02', 'scene0207_00', 'scene0256_00',
                    'scene0278_01', 'scene0304_00', 'scene0316_00', 'scene0329_02', 'scene0334_01', 'scene0342_00',
                    'scene0355_00', 'scene0356_02', 'scene0357_01', 'scene0377_01', 'scene0378_00', 'scene0406_00',
                    'scene0406_01', 'scene0414_00', 'scene0426_01', 'scene0426_03', 'scene0435_00', 'scene0435_03',
                    'scene0462_00', 'scene0488_01', 'scene0490_00', 'scene0494_00', 'scene0553_01', 'scene0553_02',
                    'scene0565_00', 'scene0574_01', 'scene0575_02', 'scene0578_00', 'scene0578_01', 'scene0598_01',
                    'scene0606_02', 'scene0607_01', 'scene0608_01', 'scene0609_02', 'scene0616_00', 'scene0629_00',
                    'scene0629_02', 'scene0633_00', 'scene0633_01', 'scene0644_00', 'scene0647_01', 'scene0648_00',
                    'scene0648_01', 'scene0651_00', 'scene0651_01', 'scene0655_00', 'scene0655_01', 'scene0660_00',
                    'scene0663_00', 'scene0663_01', 'scene0697_00', 'scene0697_01', 'scene0701_01', 'scene0702_00',
                    'scene0704_01']
    

    val_scans = ['scene0025_00', 'scene0025_02', 'scene0046_01', 'scene0222_00', 'scene0222_01', 'scene0300_00',
                'scene0412_00', 'scene0412_01', 'scene0474_01', 'scene0474_02', 'scene0474_05', 'scene0549_00',
                'scene0559_02', 'scene0595_00', 'scene0599_02', 'scene0653_01', 'scene0664_01', 'scene0671_00',
                'scene0671_01', 'scene0684_00', 'scene0700_01']

    return train_scans,val_scans


def get_ScanNet_validation_scenes():
    df = pd.read_csv('./scenes_and_contents_20_classes.csv')
    series = sorted(df.scene.tolist())
    return series


def get_experiments_and_short_names():
    a = json.load(open('../settings/experiments_and_short_names.json','r'))
    experiments = a['experiments']
    short_names = a['short_names']
    return experiments,short_names

def get_filenames():

    file_definitions = json.load(open('../settings/directory_definitions.json','r'))
    return file_definitions