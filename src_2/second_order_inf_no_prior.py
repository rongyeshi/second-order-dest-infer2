#!/usr/bin/env python

import os
import sys
import requests
import numpy as np
from itertools import groupby
import os.path
from datetime import date
import bisect
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random
import copy


a=np.mat('1,2,3;4,5,6')
b=np.array([[1,2,3],[4,5,6]])
null=[]
RouteInx=[	"	200	",	"	201	",	"	202	",	"	203	",	"	204	",	"	205	",	"	206	",	"	207	",
	"	208	",	"	209	",	"	300	",	"	301	",	"	302	",	"	303	",	"	304	",	"	305	",	"	400	",
	"	401	",	"	402	",	"	500	",	"	501	",	"	502	",	"	503	",	"	504	",	"	505	",	"	506	",
	"	507	",	"	508	",	"	600	",	"	601	",	"	602	",	"	603	",	"	604	",	"	700	",	"	V94	",
	"	701	",	"	702	",	"	703	",	"	704	",	"	705	",	"	706	",	"	707	",	"	800	",	"	801	",
	"	803	",	"	804	",	"	805	",	"	806	",	"	900	",	"	901	",	"	902	",	"	903	",	"	904	",
	"	905	",	"	906	",	"	907	",	"	103	",	"	104	",	"	1	",	"	18	",	"	22	",	"	1M	",
	"	3M	",	"	4M	",	"	5M	",	"	7M	",	"	8M	",	"	9M	",	"	10M	",	"	11M	",	"	12M	"]
    
#remove the whitespace in string fromt front and back
RouteInx=[el.strip() for el in RouteInx]


''''

'''

bound_left = -8.661915
bound_right = -8.559543
bound_top = 41.185110
bound_bot = 41.136044


#define bus info extraction function




def euclidean(vector1, vector2):
    '''calculate the euclidean distance, no numpy
    input: numpy.arrays or lists
    return: euclidean distance
    '''
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist




# example
#510
#['801', 'ATSR1', '11', 510, 35866, 37868]
#['804', 'SPC1', '11', 510, 36186, 38182]
#['804_rev', 'COVH4', '11', 510, 46754, 47024]






    




def in_step_Off_distr_dict(in_step_distr_dict,in_step_inference,timeSeg,timeTag):
    
    first_row = in_step_inference[0]
    d = len(first_row)
    num = len(in_step_inference)
    
    Off_distr_dict = in_step_distr_dict
    
    for i in range(num):
        
        route_ID = []
        seg = []
        
        ele_record = in_step_inference[i]#5D data example:    ['61148' '302' 'C24A3' 'CMO' '0' '6']
        record_time = int(ele_record[0])
        route_ID = ele_record[1]
        inx_on = int(ele_record[4])
        inx_off = int(ele_record[5])
        
        
        seg=bisect.bisect_left(timeSeg, record_time)
        tag=timeTag[seg-1]
        
        temp_array=[]
        key_name = route_ID+'_'+tag
        temp_array = Off_distr_dict[key_name]
        temp_array2 = np.copy(temp_array)
        
        temp_array2[inx_on][inx_off] +=1
        Off_distr_dict.update({key_name:temp_array2})
        
        
    
    return Off_distr_dict










def sample_index(distr_list):
    epsilon = random.uniform(0, 1)
    point = epsilon * np.sum(distr_list)
    
    if sum(distr_list)>0:
        cumulative = 0
        index =0
        for i in range(len(distr_list)):
            cumulative += distr_list[i]
            if point <= cumulative:
                index = i
                break
        return index
    else:
        return []
    





#in_step_inference, still_unknow_data, fail_count, remain_point = in_step_fill_destination(in_step_distr_dict,remain_point,OnLineStruct,selected_volume)


def in_step_fill_destination(in_step_distr_dict,incomplete_data_fold,OnLineStruct,selected_volume):
    
    
    Off_distr_dict = in_step_distr_dict
    
    first_row = incomplete_data_fold[0] #5D data example:    ['45865' '207' 'CMP1' 'JM1' '0' '16']
    d = len(first_row)
    num = len(incomplete_data_fold)
    print(d,num)
    print(first_row)
    
    timeSeg = range(0,241)#[0,4,8,12,16,20,24]
    timeTag = [str(i-1)+'-'+str(i) for i in timeSeg[1:]]
    timeSeg = [i * 360 for i in timeSeg]
    timeSeg[0]=-1
    
    fail_count = 0
    DataOnWed_train_filled_3Dto5D = []
    still_unknow_data = []
    filleSet_w_prob = []
    
    half_bw = 20*360
    
    for i in range(num):
        ele_record = incomplete_data_fold[i]
        record_time = int(ele_record[0])
        route_ID = ele_record[1]
        inx_on = int(ele_record[4])
        
        
        
        if record_time - half_bw<0:
            seg_left = 1
            tag_left = timeTag[seg_left-1]
            #seg_right = bisect.bisect_left(timeSeg, record_time + half_bw)
            #tag_right = timeTag[seg_right-1]
        else:
            seg_left = bisect.bisect_left(timeSeg, record_time - half_bw)
            tag_left = timeTag[seg_left-1]
            
        
        if record_time + half_bw>86400:
            #seg_left = bisect.bisect_left(timeSeg, record_time - half_bw)
            #tag_left = timeTag[seg_left-1]
            seg_right = 240
            tag_right = timeTag[seg_right-1]
        else:
            seg_right = bisect.bisect_left(timeSeg, record_time + half_bw)
            tag_right = timeTag[seg_right-1]
        
        
        
        route_len=len(OnLineStruct[route_ID])
        accumulate_distr_list = np.zeros((1,route_len))
        accumulate_distr_list = accumulate_distr_list[0]
        for piece in range(seg_left-1,seg_right):
            tag=timeTag[piece]
            key_name = route_ID+'_'+tag
            temp_distr = Off_distr_dict[key_name][inx_on,:]
            accumulate_distr_list = np.add(accumulate_distr_list, temp_distr)
        
        
        accumulate_distr_list = accumulate_distr_list.tolist()
        inf_off = accumulate_distr_list.index(max(accumulate_distr_list))
        
        
        
        
        if accumulate_distr_list[inf_off] != 0:
            dest_name = OnLineStruct[route_ID][inf_off]
            temp_filled = [record_time, route_ID, ele_record[2],dest_name, inx_on, inf_off,ele_record[3],ele_record[5]] #true destination name and index are the last two
            probability = accumulate_distr_list[inf_off]*1.0/sum(accumulate_distr_list)
            filleSet_w_prob.append([temp_filled,probability])
            #DataOnWed_train_filled_3Dto5D.append(temp_filled)
            
        else:
            fail_count += 1
            #still_unknow_data.append(ele_record.tolist())
            still_unknow_data.append([record_time, route_ID, ele_record[2],ele_record[3],inx_on,ele_record[5]])#['45865' '207' 'CMP1' 'JM1' '0' '16']
    
          
    filleSet_w_prob.sort(key=lambda x: x[1], reverse=True)
    
    for j in range(len(filleSet_w_prob)):
        temp_record = filleSet_w_prob[j]
        if j<selected_volume-1:
            DataOnWed_train_filled_3Dto5D.append(temp_record[0])
        else:
            still_unknow_data.append(temp_record[0])
        
        
    
    
    inferred_data = DataOnWed_train_filled_3Dto5D
    remain_point = still_unknow_data
    
    return inferred_data, still_unknow_data, fail_count, remain_point
    
    
  
def Off_distr_dict():
    """
    The off_distr_dict is a dictionary in the format of {routeID_hour1:off_distr_matrix, routeID_hour2:off_distr_matrix,...}
    The off_distr_matrix count the off bus record at a certain hour of a route, the row is the boarding bus stop index, and the column is the off bus stop index
    the hour1 hour2.... are denoted as '0-1', '1-2', ..., '23-24'.
    we have 112*24 = 2688 keys in total
    
    5D data example
    ['61148' '302' 'C24A3' 'CMO' '0' '6']
    """
    busInfo=np.load('../BusID_routeinfo_STCP.npz')
    Bus_Info_All=busInfo['Bus_Info_All']
    OnLineStruct = {}
    for ele in Bus_Info_All:
        OnLineStruct.update({ele['name']:ele['sequence']})
    
    
    
    DataOnWed = np.load('DataOnWed_coded.npz')
    DataOnWed_train_5D = DataOnWed['DataOnWed_train_5D'][0]
    
    first_row = DataOnWed_train_5D[0]
    d = len(first_row)
    num = len(DataOnWed_train_5D)
    
    
    #===================================initial the Off_distr_dict with all zaro matrix
    routeNameList_train = []
    zero_matrix_for_route = {}
    for i in range(num):
        ele_record = DataOnWed_train_5D[i]
        if ele_record[1] not in routeNameList_train:
            routeNameList_train.append(ele_record[1])
            route_len = len(OnLineStruct[ele_record[1]])
            temp_matrix = np.zeros((route_len,route_len))
            temp_matrix = temp_matrix
            zero_matrix_for_route.update({ele_record[1]:temp_matrix})
            
    
    
    Off_distr_dict = {}
    
    timeSeg = range(0,241)#[0,4,8,12,16,20,24]
    timeTag = [str(i-1)+'-'+str(i) for i in timeSeg[1:]]
    timeSeg = [i * 360 for i in timeSeg]
    timeSeg[0]=-1
    
    
    for ele_route in routeNameList_train:
        temp_dict = {}
        for ele_hour in timeTag:
            ppt = zero_matrix_for_route[ele_route]
            temp_dict.update({ele_hour:ppt})
            
            key_name = ele_route+'_'+ele_hour
            
            Off_distr_dict.update({key_name:ppt})
    
    
    
    #============================count the records in the matrix framework for sampling purpose
    
    for i in range(num):
        #print(i)
        route_ID = []
        seg = []
        
        ele_record = DataOnWed_train_5D[i]
        record_time = int(ele_record[0])
        route_ID = ele_record[1]
        inx_on = int(ele_record[4])
        inx_off = int(ele_record[5])
        
        
        seg=bisect.bisect_left(timeSeg, record_time)
        tag=timeTag[seg-1]
        
        temp_array=[]
        key_name = route_ID+'_'+tag
        temp_array = Off_distr_dict[key_name]
        temp_array2 = np.copy(temp_array)
        
        temp_array2[inx_on][inx_off] +=1
        Off_distr_dict.update({key_name:temp_array2})
        
        
    np.savez('Off_distr_dict_6min_ini.npz', Off_distr_dict=[Off_distr_dict])


    
def inference_validate():
    busInfo=np.load('../BusID_routeinfo_STCP.npz')
    Bus_Info_All=busInfo['Bus_Info_All']
    
    OnLineStruct = {}
    for ele in Bus_Info_All:
        OnLineStruct.update({ele['name']:ele['sequence']})
    #print(OnLineStruct['502'])
    
    DataOnWed = np.load('DataOnWed_coded.npz')
    DataOnWed_test_5D = DataOnWed['DataOnWed_test_5D'][0]
    
    Off_distr_dict = np.load('Off_distr_dict_6min_ini.npz')
    Off_distr_dict = Off_distr_dict['Off_distr_dict'][0]
    original_dict = Off_distr_dict.copy()
    
    first_row = DataOnWed_test_5D[0] #5D data example:    ['45865' '207' 'CMP1' 'JM1' '0' '16']
    d = len(first_row)
    num = len(DataOnWed_test_5D)
    
    
    timeSeg = range(0,241)#[0,4,8,12,16,20,24]
    timeTag = [str(i-1)+'-'+str(i) for i in timeSeg[1:]]
    timeSeg = [i * 360 for i in timeSeg]
    timeSeg[0]=-1
    
    
    
    fail_count = 0
    DataOnWed_test_inf_for_recon = []
    error_inf =0
    error_rand = 0
    
    #==========================initialize recon_data_set and incomplete_dataset===========================================
    
    
    in_step_distr_dict = Off_distr_dict # initialize the distribution disctionary
    #in_step_recon_data = DataOnWed['DataOnWed_train_5D'][0]#5D data example:    ['45865' '207' 'CMP1' 'JM1' '0' '16']
    in_step_incomplete_data = DataOnWed['DataOnWed_test_5D'][0]
    
    in_step_inference = []
    inferenced_chunk = []
    num = len(in_step_incomplete_data)
    
    
    iter_num = 200#10#1#50
    fold_size = int(num/iter_num)
    
    folds = []
    for i in range(iter_num):
        if (i+1) != iter_num:
            temp_fold = in_step_incomplete_data[i*fold_size:(i+1)*fold_size,:]
        
        if (i+1) == iter_num:
            temp_fold = in_step_incomplete_data[i*fold_size:,:]
        folds.append(temp_fold)
    
    
    
    
    
    selected_point = folds[0]#initial
    remain_point = in_step_incomplete_data
    selected_volume = fold_size
    iter = 0
    
    
    
    #===========================iteration start================================
    
    while len(selected_point) >= selected_volume-1:
        
        #==============update distribution dictionary w. newly inferred data fold
        
        
        
        if iter>0:
            in_step_distr_dict = in_step_Off_distr_dict(in_step_distr_dict,in_step_inference,timeSeg,timeTag)
            
            
        #==============with the updated distribution dictinary, infer next fold
        
        in_step_inference, still_unknow_data, fail_count, remain_point = in_step_fill_destination(in_step_distr_dict,remain_point,OnLineStruct,selected_volume)
        selected_point = in_step_inference
        inferenced_chunk+=in_step_inference
        
        
        remain_point = np.asarray(remain_point+[])
        
        print('Iteration:',iter,len(inferenced_chunk),len(inferenced_chunk)*1.0/num)
        iter+=1
        
        error_inf=0
        true_count = 0
        for reconstruct in inferenced_chunk:#5D data example:    ['45865' '207' 'CMP1' 'JM1' '0' '16', true-off-name, true-off-inx]
            inf_off = int(reconstruct[5])
            true_off= int(reconstruct[7])
            
            error_inf += (inf_off-true_off)**2
            if inf_off ==true_off:
                true_count+=1
        print('error_inf:',(1.0*error_inf)/len(inferenced_chunk))
        print('accuracy (inside inferred set):',(1.0*true_count)/len(inferenced_chunk), 'total_accuracy_inf (include both labeled and unlabeled set):', (1.0*true_count)/num)
            
    
    
    
    
    
    
    
    
    
    






if __name__ == "__main__":
    Off_distr_dict()
    print('saved Off_distr_dict_6min.npz')
    print('Alighting distribution in every 6 minutes time window')
    print('=====================step Three complete==================')
    
    inference_validate()
    print('Validate the second order inference accuracy')
    print('=====================step Four complete==================')
    
    
        
    

