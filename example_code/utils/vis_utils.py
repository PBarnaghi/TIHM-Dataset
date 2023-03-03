import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os 
from matplotlib import cm
from joypy import joyplot
import matplotlib.patches as mpatches
from collections.abc import Iterable

from .data_utils import *

def save_fig(fname,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path,fname),bbox_inches='tight')

def save_personal_fig(pid,fname,save_path):
    save_path = os.path.join(save_path,pid)
    save_fig(fname,save_path)

def add_personal_label_markers(labels_df,pid,ax,max_v,add_marker=True,marker_color='orchid',
                               add_lgd=True,lgd_loc=[1.01,0.75],min_v=0):

    lbs = labels_df.loc[labels_df.patient_id==pid]
    lmarkers = ['*','s','^','o','P','p','h','D']
    lax, lax_lgd = [], []
    
    if add_marker:
        for i,l in enumerate(lbs.type.values.categories):
            ltmp = lbs.loc[lbs.type==l]
            if len(ltmp)>0:
                ll = ax.plot(ltmp['date'],[max_v]*len(ltmp),lmarkers[i],color=marker_color)
                lax.append(ll[0])
                lax_lgd.append(l)
                
    for i in range(len(lbs)):
        ax.plot([lbs.date.values[i],lbs.date.values[i]],[min_v,max_v],'--',color='C7')
        
    if add_lgd:
        lb_patch = mpatches.Patch(color='orchid')
        lbl_lgd = ax.legend([lb_patch]+lax, ['Alerts']+lax_lgd, loc=lgd_loc,fontsize=9) 
        ax.add_artist(lbl_lgd)
        return None, None
    else:
        return lax,lax_lgd
    

def dsample_xticks(N,m=10):
    gap = N/m    
    idx = [int(i*gap) for i in range(m)]
    if N%m>0:
        idx+=[N-1]
    return idx
    

def draw_day_ridge_plot(day_df,ax=None,xticks=None):
    x_range = np.arange(day_df.shape[0])
    f, axes = joyplot(day_df,kind="values",x_range=x_range,overlap = 0.5, colormap=cm.Blues_r,
                      fade=True,figsize=[10,4],linewidth=0.25,alpha=0.6,ax=ax)

    if xticks is None:
        idx = dsample_xticks(day_df.shape[0],m=10)
        
        axes[-1].set_xticks(x_range[idx],labels=day_df.index[idx])
    else:
        axes[-1].set_xticks(xticks[0],xticks[1])
    return
    
    
def vis_day_counts_ridge_plot(raw_df,col,title,fname,save_path='./figs/',drop_list=None,
                              transform=False,ax=None,xticks=None):    
    if transform:
        tmp = raw_df.groupby([col, 'date']).size().to_frame('size').reset_index().pivot_table(values = 'size', columns=col, index='date')
    else:
        tmp = raw_df.groupby('date').agg('sum')
    if drop_list is not None:
        tmp = tmp.drop(drop_list,axis=1)
    draw_day_ridge_plot(tmp,ax=ax,xticks=xticks)
    plt.title(title)
    save_fig(fname,save_path=save_path)
    
    
def get_personal_sleep_day_df(sleep_df,pid):
    
    sleep_states = ['AWAKE', 'LIGHT', 'DEEP', 'REM']
    sleep_df = sleep_df.set_index('date')
    sdtmp = get_personal_df(sleep_df,pid).loc[:,sleep_states]#get_personal_agg_info_df(sleep_df,pid,cols=sleep_states,keys=['date'],agg_fn='sum') 
    pdtmp = get_personal_df(sleep_df,pid).loc[:,['heart_rate','respiratory_rate']] #get_personal_agg_info_df(sleep_df,pid,cols=['heart_rate','respiratory_rate'],keys=['date'],agg_fn='max')

    return sdtmp,pdtmp
    
def vis_personal_aligned_multiview_day_plot(phys_df,pgrps,pid,save_path,sleep_df=None,act_df=None,
                                            labels_df=None,ftype='svg',xticks=None):
    
    sns.set_style("whitegrid",{'axes.grid': True,'grid.linestyle': '--', 'axes.spines.left':False,
                            'axes.spines.right':False,'axes.spines.top':False})
    cmap = sns.color_palette('colorblind')#cm.get_cmap('tab10').colors

    #sleep_states = sleep_df['state'].unique()
    if sleep_df is not None:
        sdtmp, _ = get_personal_sleep_day_df(sleep_df,pid)
    if act_df is not None:
        adtmp = get_personal_df(act_df,pid)
        adtmp = adtmp.set_index('date')
        adtmp['total'] = adtmp.sum(axis=1)
        
    if sleep_df is not None and len(sdtmp) > 0 and act_df is not None:
        adtmp,sdtmp = align_dates(adtmp,sdtmp)

        
    
    pdtmp = get_personal_df(phys_df,pid)
    #pdtmp = pdtmp.set_index('date')
    if act_df is not None:
        pdtmp.drop(pdtmp.loc[(pdtmp.date<adtmp.index.values.min())|(pdtmp.date>adtmp.index.values.max())].index,axis=0,inplace=True)
    pv = pd.pivot_table(pdtmp, values='value', index=['date'],
                    columns=['device_type'])
    pv = pv.rename(columns={'Skin Temperature':'Skin temperature','Body Temperature':'Body temperature'})
    pvs = []
    for i,pgrp in enumerate(pgrps.values()):
        pgrp = (set(pgrp) & set(pv.columns))    
        if len(pgrp)>0:
            pvi = pv[pgrp]
            pvi = pvi.drop(pvi.columns[pvi.values.sum(axis=0)==0],axis=1)
            pvs.append(pvi)

    
    
    nr = len(pvs)+1 if act_df is not None else len(pvs)
    f, axs = plt.subplots(nr, 1,gridspec_kw={'height_ratios': [1]*nr},figsize=[10,3*nr],sharex=True)
    
    ci = 0
    axi = 0
    if act_df is not None:
        aax = axs[0].fill_between(x=adtmp.index,y1=np.zeros(adtmp.shape[0]),y2=adtmp['total'].values,alpha=0.4,color=cmap[ci])
        handles = [aax]
        max_v = adtmp.values.max()
        lgd = ['Acitivity counts']
        ci+=1
        if sleep_df is not None and len(sdtmp) > 0:
            sdtmp['total'] = sdtmp.sum(axis=1)            
            sax = axs[0].fill_between(x=sdtmp.index,y1=np.zeros(sdtmp.shape[0]),y2=sdtmp['total'].values,alpha=0.4,color=cmap[ci]) 
            ci+=1   
            handles.append(sax)
            max_v = max(max_v,sdtmp.values.max())
            lgd.append('Sleep duration (minutes)')
            
        axs[0].tick_params(labelbottom=True)   
        axi += 1

    if labels_df is not None:
        if act_df is not None:
            labels_df = labels_df.drop(labels_df.loc[(labels_df.date<adtmp.index.values.min())|(labels_df.date>adtmp.index.values.max())].index,axis=0)
        labels_df = labels_df.loc[labels_df['patient_id']==pid]
        if len(labels_df)==0:
            labels_df = None

    if labels_df is not None and axi >0:
        add_personal_label_markers(pid=pid,ax=axs[0],max_v=max_v,
                                   labels_df=labels_df,lgd_loc=[1.01,0.5],add_lgd=True)      
        axs[0].legend(handles,lgd,loc=[1.01,0.1]) 

        

    for i,pvi in enumerate(pvs):
        
        lls = axs[i+axi].plot(pvi.index,pvi.values,'.-')
        for l in lls:           
            l.set_color(cmap[ci])
            ci+=1
        
        axs[i+axi].tick_params(labelbottom=True)
        
        max_v = pvi.fillna(0).values.max()*1.1
        min_v = pvi.fillna(999).values.min()*0.9
        
        if labels_df is not None:
            if i+axi>0:
                add_personal_label_markers(pid=pid,ax=axs[i+axi],max_v=max_v,
                                       labels_df=labels_df,add_marker=True,add_lgd=False,min_v=min_v) 
            else:
                add_personal_label_markers(pid=pid,ax=axs[0],max_v=max_v,
                                   labels_df=labels_df,lgd_loc=[1.01,0.5],add_lgd=True)      
  
        axs[i+axi].legend(pvi.columns,loc=[1.01,0.1])
    
    if xticks is None:
        if act_df is None:
            xidx = dsample_xticks((pvs[0].index[-1]-pvs[0].index[0]).days,m=8) 
            xticks = [pvs[0].index.values[0]+pd.Timedelta(xi,'D') for xi in xidx]
        else:
            xidx = dsample_xticks((adtmp.index[-1]-adtmp.index[0]).days,m=8) 
            xticks = [adtmp.index.values[0]+pd.Timedelta(xi,'D') for xi in xidx]
        axs[0].set_xticks(xticks,labels=xticks)
        
    else:
        axs[0].set_xticks(xticks[0],labels=xticks[1])
    
    #axs[0].set_title('Patient '+str(pid)+' aligned multiview day trend')
    save_personal_fig(pid=pid,fname='personal_multiview_day_plot.'+ftype,save_path=save_path) 
    
    

 