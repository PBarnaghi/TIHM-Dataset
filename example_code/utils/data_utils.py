import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def concat_slist(slist):
    s = ''
    for vs in slist:
        s+= str(vs)+',' 
    return s

def correct_col_type(df,col):
    raw_type = str(type(df[col].dtype)).split('.')[-1].split('\'')[0]
    #print(col,raw_type)
    if 'object' in raw_type:
        if 'date' in col:
            return pd.to_datetime(df[col])
        else:
            return df[col].astype('category')
    else:
        return df[col]
    
    
def gen_date_col(df, tcol):
    df['date'] = df[tcol].dt.date
    return df

def transform_category_to_counts(df,col,keys):
    tmp = df.groupby([col]+ keys).size().to_frame('size').reset_index().pivot_table(values = 'size', columns=col, index=keys)
    tmp = tmp.drop(tmp.index[tmp.values.sum(axis=1)==0],axis=0).reset_index()
    return tmp


def get_personal_df(df,pid):
    if not 'patient_id' in df.columns:
        df = df.reset_index()
    tmp = df.loc[df.patient_id==pid].drop('patient_id',axis=1)
    
    return tmp



def min_max_perpatient(df,skip=[]):
    for pid in df.patient_id.unique():
        ptmp = df.loc[df.patient_id==pid]
        for c in ptmp.columns:
            if 'int' in str(ptmp[c].dtype) or 'float' in str(ptmp[c].dtype):
                if ptmp[c].notna().sum() > 0 and c not in skip:
                    min_v = np.nanmin(ptmp[c].values)
                    max_v = np.nanmax(ptmp[c].values)
                    if max_v > min_v:
                        df.loc[df.patient_id==pid,c] = (ptmp[c].values-min_v)/(max_v-min_v)
                    elif max_v!=0:
                        df.loc[df.patient_id==pid,c] = 0.5 #only one record 
    return df   


def gen_summary(df):
    sm = pd.DataFrame(columns=['Value Type','Value Number','Description'])
    for stc in df.columns:
        sm.loc[stc,'Value Type'] = str(type(df[stc].dtype)).split('.')[-1].split('\'')[0]
        if 'Categorical' in sm.loc[stc,'Value Type'] or 'object' in sm.loc[stc,'Value Type']:
            vset = set(df[stc].values)
            sm.loc[stc, 'Value Number'] = len(vset)
            dl = len(vset) if 5 > len(vset) else 5
            if 'id' in stc:
                sm.loc[stc,'Description'] = 'hash code'
            else:
                sm.loc[stc,'Description'] = concat_slist(list(vset)[:dl])
        elif 'datetime' in sm.loc[stc,'Value Type'].lower(): 
            sm.loc[stc,'Description'] = 'from '+ str(df[stc].min()) + ' to ' + str(str(df[stc].max()))
        elif 'float' in sm.loc[stc,'Value Type'] or 'int' in sm.loc[stc,'Value Type']: 
            sm.loc[stc,'Description'] = 'min: ' + str(df[stc].min()) + ', max: ' + str(str(df[stc].max()))
        elif 'bool' in sm.loc[stc,'Value Type']: 
            sm.loc[stc, 'Value Number'] = 2
            sm.loc[stc,'Description'] = 'True or False'

    return sm             

