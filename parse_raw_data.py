import numpy as np
import pandas as pd
import sys
import os

import matplotlib.pyplot as plt
import re
from os import walk

def parce_fileName(fileName):
    tmp = re.split('\.', fileName)
    file_date_code = tmp[0]
    tmp = re.split('-', file_date_code)
    code = int(tmp[0])
    date_str = "-".join(tmp[1:])
    file_date = pd.to_datetime(date_str, format="%Y-%m-%d")
    return code, file_date

def obtain_raw_data():
    #dir_path = 'C:\Users\Mikhail\Documents\MobCPA\data\lead_data\lead_data_up_to31may'

    # global parameters
    dir_path = '.\lead_data\lead_data_up_to31may'
    dir_data_csv = '.\data_csv'
    code_check = 235


    # obtain all files in the dir and parse the name
    fileList = []
    for (dirpath, dirnames, filenames) in walk(dir_path):
        fileList.extend(filenames)
        break

    codes = list()
    file_dates = list()
    wns = list()
    for f in fileList:
        code, file_date = parce_fileName(f)
        codes.append(code)
        file_dates.append(file_date)
        if code == code_check:
            wns.append(f)
            # print code , file_date , f

    # remove duplicates
    wns = set(wns)
    wns = list(wns)

    codes = list()
    file_dates = list()

    for f in wns:
        code, file_date = parce_fileName(f)
        codes.append(code)
        file_dates.append(file_date)

    fileInfo = pd.DataFrame({'CountryCode': codes, 'FileDate': file_dates, 'WholeName': wns})
    # print fileInfo.head(5)

    # choose required
    uniq_codes = list(set(fileInfo.CountryCode))  # here we have 4 unique code : 0, 1, 2, 3
    idx_choose = 0
    # idxs_code = fileInfo.loc[fileInfo.CountryCode == uniq_codes[idx_choose]]

    uniq_code_use = uniq_codes[idx_choose]
    str_file_name = "".join(['data_cut_file_', str(uniq_code_use), '.txt'])  # in that file we load final data

    idxs_code = fileInfo.loc[:, 'CountryCode'] == uniq_code_use
    fileInfo_ucode = fileInfo.loc[idxs_code, :]  # list all files with specific unique code


    # handle bad intervals
    # assign bad intervals
    bad_list = list()

    if True:
        date1_str = '2017-04-20'
        date2_str = '2017-05-2'
        date1 = pd.to_datetime(date1_str, format='%Y-%m-%d')
        date2 = pd.to_datetime(date2_str, format='%Y-%m-%d')
        # print date1 , date2
        bad_int = (date1, date2)
        bad_list.append(bad_int)

    if True:
        date1_str = '2017-05-17'
        date2_str = '2017-05-17'
        date1 = pd.to_datetime(date1_str, format='%Y-%m-%d')
        date2 = pd.to_datetime(date2_str, format='%Y-%m-%d')
        bad_int = (date1, date2)
        bad_list.append(bad_int)

    # exclude last day
    if True:
        date1_str = '2017-05-31'
        date2_str = '2017-05-31'
        date1 = pd.to_datetime(date1_str, format='%Y-%m-%d')
        date2 = pd.to_datetime(date2_str, format='%Y-%m-%d')
        bad_int = (date1, date2)
        bad_list.append(bad_int)

    # exclude special days
    if False:  # this is outlier for code=283
        date1_str = '2017-05-17'
        date2_str = '2017-05-17'
        date1 = pd.to_datetime(date1_str, format='%Y-%m-%d')
        date2 = pd.to_datetime(date2_str, format='%Y-%m-%d')
        bad_int = (date1, date2)
        bad_list.append(bad_int)

    # find valid indexis
    k = 0
    if len(bad_list) == 0:
        idx_fin = range(fileInfo_ucode.shape[0])
    else:
        for bad_int in bad_list:
            idx_cut1 = (fileInfo_ucode.loc[:, 'FileDate'] < bad_int[0])
            idx_cut2 = (fileInfo_ucode.loc[:, 'FileDate'] > bad_int[1])

            idx_tmp = [a or b for a, b in zip(idx_cut1, idx_cut2)]
            if k == 0:
                idx_fin = idx_tmp
            else:
                idx_fin = [a and b for a, b in zip(idx_fin, idx_tmp)]
            k += 1

    # apply final indexes
    fileInfo_ucode = fileInfo_ucode.loc[idx_fin, :]
    fileInfo_ucode.index = range(len(fileInfo_ucode))

    a = fileInfo_ucode.FileDate.sort_values(axis=0, ascending=True)
    fileInfo_ucode = fileInfo_ucode.iloc[a.index.values, :]
    fileInfo_ucode.index = range(len(fileInfo_ucode))

    idxs = list(fileInfo_ucode.index)  # set of valid files

    # preparation for storage
    deltaMax = 300  # maximal interval of revenue
    arr_total = np.zeros((0, deltaMax))  # array for each delta revenue
    rsum_total = np.zeros((0, 1))  # total sum for each file
    all_dates = list()  # date of files
    all_code = list()  # code of files

    fig_idx = 0

    if True:

        for idx in idxs:

            fileName = fileInfo_ucode.WholeName[idx]
            file_date = fileInfo_ucode.FileDate[idx]
            code = fileInfo_ucode.CountryCode[idx]

            all_dates.append(file_date)
            all_code.append(code)

            print "Parce file with code={} and date={}".format(code, file_date)

            path_to_file = os.path.join(dir_path, fileName)
            data_tmp = pd.read_csv(path_to_file, sep=';', header=None)
            data_tmp.columns = ['date', 'rev', 'leadnum']
            data_tmp['date'] = data_tmp[['date']].apply(pd.to_datetime, format='%d.%m.%Y')

            a = data_tmp.date.sort_values(axis=0, ascending=False)
            data_sort = data_tmp.iloc[a.index.values, :]
            data_sort.index = range(len(data_sort))

            # cut future wrong date
            idx_cut = data_sort.loc[:, 'date'] <= file_date
            data_cut = data_sort.loc[idx_cut, :]
            data_cut.index = range(len(data_cut))

            # cut special periods

            # handle bad intervals
            # find valid indexis
            if False:
                k = 0
                for bad_int in bad_list:
                    idx_cut1 = (data_cut.loc[:, 'date'] < bad_int[0])
                    idx_cut2 = (data_cut.loc[:, 'date'] > bad_int[1])

                    idx_tmp = [a or b for a, b in zip(idx_cut1, idx_cut2)]
                    if k == 0:
                        idx_fin = idx_tmp
                    else:
                        idx_fin = [a and b for a, b in zip(idx_fin, idx_tmp)]
                    k += 1

                # apply final indexes
                data_cut = data_cut.loc[idx_fin, :]
                data_cut.index = range(len(data_cut))

            # create new column deltaT
            date_tmp = data_cut.date
            deltaT = [(date_tmp[0] - date_tmp[idx]).days for idx in range(date_tmp.shape[0])]
            # print deltaT

            data_fin = data_cut[:]
            data_fin['DeltaDays'] = deltaT

            # print data_fin.head(5)
            # print data_fin.shape[0]

            # store revenue from this file in final array
            arr_rev = np.zeros((1, deltaMax))
            sum_r = 0
            for r, d in zip(data_fin.rev, data_fin.DeltaDays):
                if d < deltaMax:
                    arr_rev[0, d] = r
                sum_r += r

            arr_total = np.vstack((arr_total, arr_rev))
            sum_r = np.reshape(sum_r, (1, 1))
            rsum_total = np.vstack((rsum_total, sum_r))

            # dir_path_fig = 'C:\Users\Mikhail\Documents\MobCPA\ipython_scr\lead_cut_prj\RawData_new_fig_{}'.format(code)
            dir_path_fig = '.\Raw_figs\Raw_figs_{}'.format(code)

            if not os.path.exists(dir_path_fig):
                os.makedirs(dir_path_fig)

            cut_date, day_time = re.split(' ', str(file_date))
            figName = "Raw_dat_code_{}_date_{}.png".format(code, cut_date)
            path_to_fig = os.path.join(dir_path_fig, figName)

            if True:
                fig = plt.figure(fig_idx)
                plt.plot(data_fin.DeltaDays, data_fin.rev, 'r-')
                plt.axvline(30, color='b')
                plt.axvline(60, color='b')
                plt.xlabel('Delta Days')
                plt.ylabel('Revenue')
                title_name = 'Code={} date={}'.format(code, cut_date)
                plt.title(title_name)
                plt.grid()
                fig.savefig(path_to_fig, bbox_inches='tight')
                plt.pause(0.5)
                #plt.show()
                fig_idx += 1
                # raw_input()

        data_all = pd.DataFrame(arr_total, index=range(arr_total.shape[0]))
        data_all['sum_rev'] = rsum_total
        data_all['Dates'] = all_dates
        data_all['Codes'] = all_code

        path_to_csv = os.path.join(dir_data_csv, str_file_name)
        data_all.to_csv(path_to_csv, index=range(data_all.shape[0]))

    return fileInfo_ucode
