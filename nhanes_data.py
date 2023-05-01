import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

YEARS = [2009, 2011, 2013]
PRIMARY_KEY = "SEQN"
GENDER = "RIAGENDR"
AGE = "RIDAGEYR"
RACE = "RIDRETH1"
RACE_CODES = {1:"MEX_AMER", 2:"OTHER_HISP", 3:"WHITE", 4:"BLACK", 5:"OTHER"}

VITAMIN_B12 = "LBDB12"

TTG_KEY = 'LBXTTG'
EMA_KEY = 'LBXEMA'

CREAT_PHOSPHO = "LBXSCK"
CELIAC = "MCQ082" # Ever been told you have celiac disease? 1 = yes 2 = no
GF_DIET = "MCQ086" # Are you on a gluten-free diet?
DIABETES = "DIQ010"

WEIGHT = "BMXWT"
HEIGHT = "BMXHT"

# Set of keys for biochem blood tests
# Note that LBXSCK (creatine phosphokinase) has lower sample size
BIOCHEM_KEYS = ['LBXSAL', 'LBDSALSI', 'LBXSATSI', 'LBXSASSI', 'LBXSAPSI',
       'LBXSBU', 'LBDSBUSI', 'LBXSCA', 'LBDSCASI', 'LBXSCH', 'LBDSCHSI',
       'LBXSC3SI', 'LBXSCR', 'LBDSCRSI', 'LBXSGTSI', 'LBXSGL', 'LBDSGLSI',
       'LBXSIR', 'LBDSIRSI', 'LBXSLDSI', 'LBXSPH', 'LBDSPHSI', 'LBXSTB',
       'LBDSTBSI', 'LBXSTP', 'LBDSTPSI', 'LBXSTR', 'LBDSTRSI', 'LBXSUA',
       'LBDSUASI', 'LBXSNASI', 'LBXSKSI', 'LBXSCLSI', 'LBXSOSSI', 'LBXSGB',
       'LBDSGBSI']

CBC_KEYS = ['LBXWBCSI', 'LBXLYPCT', 'LBXMOPCT', 'LBXNEPCT', 'LBXEOPCT',
 'LBXBAPCT', 'LBDLYMNO', 'LBDMONO', 'LBDNENO', 'LBDEONO', 'LBDBANO', 'LBXRBCSI',
 'LBXHGB', 'LBXHCT', 'LBXMCVSI', 'LBXMCHSI', 'LBXMC', 'LBXRDW', 'LBXPLTSI',
 'LBXMPSI']

BIOCHEM_SI = ['LBDSALSI', 'LBXSATSI', 'LBXSASSI', 'LBXSAPSI',
        'LBDSBUSI', 'LBDSCASI', 'LBDSCHSI', 'LBXSC3SI', 'LBDSCRSI', 
        'LBXSGTSI', 'LBDSGLSI', 'LBDSIRSI', 'LBXSLDSI', 'LBDSPHSI',
        'LBDSTBSI', 'LBDSTPSI', 'LBDSTRSI', 'LBDSUASI', 'LBXSNASI', 
        'LBXSKSI', 'LBXSCLSI', 'LBXSOSSI', 'LBDSGBSI']
QUESTION_KEYS = ['MCQ010', 'MCQ025', 'MCQ035', 'MCQ040', 'MCQ050', 
                  'MCQ051', 'MCQ053', 'MCQ070', 'MCQ080', 'MCQ082', 
                  'MCQ086', 'MCQ092', 'MCD093', 'MCQ140', 'MCQ149', 
                  'MCQ160A', 'MCQ180A', 'MCQ191', 'MCQ160N', 'MCQ180N',
                  'MCQ160B', 'MCQ180B', 'MCQ160C', 'MCQ180C', 'MCQ160D',
                  'MCQ180D', 'MCQ160E', 'MCQ180E', 'MCQ160F', 'MCQ180F',
                  'MCQ160G', 'MCQ180G', 'MCQ160M', 'MCQ170M', 'MCQ180M',
                  'MCQ160K', 'MCQ170K', 'MCQ180K', 'MCQ160L', 'MCQ170L',
                  'MCQ180L', 'MCQ220', 'MCQ230A', 'MCQ230B', 'MCQ230C',
                  'MCQ230D', 'MCQ240A', 'MCQ240AA', 'MCQ240B', 'MCQ240BB',
                  'MCQ240C', 'MCQ240CC', 'MCQ240D', 'MCQ240DD', 'MCQ240DK',
                  'MCQ240E', 'MCQ240F', 'MCQ240G', 'MCQ240H', 'MCQ240I',
                  'MCQ240J', 'MCQ240K', 'MCQ240L', 'MCQ240M', 'MCQ240N', 
                  'MCQ240O', 'MCQ240P', 'MCQ240Q', 'MCQ240R', 'MCQ240S', 
                  'MCQ240T', 'MCQ240U', 'MCQ240V', 'MCQ240W', 'MCQ240X', 
                  'MCQ240Y', 'MCQ240Z', 'MCQ300A', 'MCQ300B', 'MCQ300C', 
                  'MCQ075', 'MCQ084', 'MCQ195', 'MCQ365A', 'MCQ365B', 
                  'MCQ365C', 'MCQ365D', 'MCQ370A', 'MCQ370B', 'MCQ370C', 
                  'MCQ370D', 'MCQ380', 'MCQ151', 'MCQ160O', 'MCQ203', 
                  'MCQ206']

CANCER = "MCQ220" # Ever told you have cancer? 1 = yes, 2 = no, others
THYROID = "MCQ160M"
LIVER = "MCQ170L"
TTG_KEYS = [TTG_KEY, EMA_KEY]

def get_sas_data(year):
    directory = "IW Data/" + str(year)
    ret = None
    for file in os.listdir(directory):
        f = os.path.join(directory, file)

        # skip if not file
        if not os.path.isfile(f): continue

        name = f
        f = pd.read_sas(f)
        if 'CBC' in name: print(f"{name}: {np.array(f.columns)}")

        # combine columns on SEQN
        if ret is None:
            ret = f
        else:
            ret = pd.merge(ret, f, on=PRIMARY_KEY)
    return ret

def get_all_sas_data(filter=None):
    years = []
    for year in YEARS:
        years.append(get_sas_data(year))
    data = pd.concat(years, ignore_index=True)

    if RACE in filter:
        frames = [pd.DataFrame({val: np.where(data[RACE] == key, 1, 0)}) for key,val in RACE_CODES.items()]
        data = pd.concat([data] + frames, axis=1)
        filter.remove(RACE)
        for val in RACE_CODES.values():
            filter.append(val)
    return filter_data(data, filter=filter)

def get_xy_data(xfilter, yfilter, modify=None, dropna=True):
    data = get_all_sas_data(filter=xfilter + yfilter)
    if RACE in xfilter:
        xfilter.remove(RACE)
        for val in RACE_CODES.values():
            xfilter.append(val)
    if RACE in yfilter:
        yfilter.remove(RACE)
        for val in RACE_CODES.values():
            yfilter.append(val)
    if modify: 
        modify(data)
    if dropna: data = data.dropna()

    return data[xfilter], data[yfilter]

def filter_data(data, filter):
    if not filter: return data        
    return data[filter]

if __name__ == "__main__":
    def modify(data):
        pass
        # data[EMA_KEY][data[EMA_KEY] != 1] = 0

        # data[GENDER] = data[GENDER] - 1
        
    a, b = get_xy_data([TTG_KEY],[EMA_KEY], modify=modify, dropna=False)
    print(a)
    print(b)
    print(a[(b[EMA_KEY] != 0) & (b[EMA_KEY])])
    print(b.value_counts())
    for col in a.columns:
        print(f"{col}: {len(a[col][pd.isna(a[col])])}")