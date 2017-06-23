# in this module all data processing functions are declared

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

import statsmodels
import scipy as sc
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.graphics.regressionplots import plot_leverage_resid2

class LeadModel:

    def __init__(self, param_dict):
        self.param_dict = param_dict

    def do_model(self):

        # global params
        code = self.param_dict.get('code') # obtain code
        deltaIR_max = 60
        self.deltaIR_max = deltaIR_max

        # try to open file with this code
        prefix = 'data_cut_file_'
        file_name = "".join([prefix, str(code), '.txt'])
        dir_path = './data_csv'
        full_path = os.path.join(dir_path, file_name)

        try:
            data = pd.read_csv(full_path, sep=',', header=0)
        except:
            print "Can not open file={}".format(file_name)
            raise

        # sort data frame by Dates
        a = data.Dates.sort_values(axis=0, ascending=True)
        data_sort_test = data.iloc[a.index.values, :]
        data_sort_test.index = range(len(data))

        # create impulse response data frame
        # form new dataframe
        hAr = np.zeros((0, deltaIR_max))  # array of impulse response
        dateIR = list()  # array of initial date
        Sum_r = np.zeros((0, 1))
        lDate = data_sort_test.shape[0]

        for dateCalc_idx in range(lDate):
            tmp_IR = np.zeros((1, deltaIR_max))
            dateCalc_pd = pd.to_datetime(data_sort_test.Dates[dateCalc_idx], format='%Y-%m-%d')
            for dateCheck_idx in range(dateCalc_idx, lDate):
                dateCheck_pd = pd.to_datetime(data_sort_test.Dates[dateCheck_idx], format='%Y-%m-%d')
                delta_time = (dateCheck_pd - dateCalc_pd).days
                # print delta_time , dateCheck_idx
                tmp_IR[0, delta_time] = data_sort_test.iloc[dateCheck_idx, delta_time + 1]
            hAr = np.vstack((hAr, tmp_IR))
            sum_r = tmp_IR.sum(axis=1)
            # print sum_r
            Sum_r = np.vstack((Sum_r, sum_r))
            dateIR.append(dateCalc_pd)

        l1 = len(dateIR)
        dateIR_np = np.reshape(dateIR, (l1, 1))

        pdIR = pd.DataFrame(hAr, index=range(hAr.shape[0]))
        pdIR['Dates'] = dateIR_np
        pdIR['Rev'] = Sum_r

        # here possible processing over obtained impulse response
        self.pdIR = pdIR
        test_df = self.create_features()
        return test_df

    def create_features(self):
        pdIR = self.pdIR
        Thr_list = self.param_dict.get('Thr_hist')
        nu = self.param_dict.get('nu')
        self.nu = nu

        Xhist_test = np.zeros((pdIR.shape[0], 0))  # array of new features
        thr0 = 0
        for thr in Thr_list:
            NewFeat = np.sum(pdIR.iloc[:, thr0:thr], axis=1)
            NewFeat = np.reshape(NewFeat, (pdIR.shape[0], 1))
            Xhist_test = np.hstack((Xhist_test, NewFeat))
            thr0 = thr

        y_gross = pdIR.Rev
        y_return = y_gross * nu

        Dates = pdIR.Dates
        Date_str = [a.strftime('%Y-%m-%d') for a in Dates]
        X_all = pdIR.iloc[:, :self.deltaIR_max]

        # create initial OLS analysis
        if Xhist_test.shape[1] == 2:
            pd_test = pd.DataFrame(Xhist_test, columns=['F1', 'F2'])
        if Xhist_test.shape[1] == 3:
            pd_test = pd.DataFrame(Xhist_test, columns=['F1', 'F2', 'F3'])
        pd_test['Rev'] = y_return

        if Xhist_test.shape[1] == 2:
            m1 = smf.ols('Rev ~ F1 + F2', data=pd_test)
        if Xhist_test.shape[1] == 3:
            m1 = smf.ols('Rev ~ F1 + F2 + F3', data=pd_test)

        fitted = m1.fit()
        # end OLS analysis

        # apply Ridge model
        self.Xhist = Xhist_test
        self.calc_Ridge()

        code = self.param_dict.get('code')
        dir_path = './Model_rez_{}'.format(code)
        file_name = 'OLS_summary_{}.txt'.format(code)
        full_path = os.path.join(dir_path, file_name)
        file_tmp = open(full_path, 'w')
        file_tmp.write(fitted.summary().as_text())
        file_tmp.close()

        return fitted.summary()

    def calc_Ridge(self):

        pdIR = self.pdIR
        deltaIR_max = self.deltaIR_max
        nu = self.nu
        numF_ir = self.param_dict.get('numF_ir')
        seed_use = 42
        calc_b0 = False
        alpha_use = 0.1
        doSave = True
        code = self.param_dict.get('code')
        Thr_list = self.param_dict.get('Thr_hist')

        if self.param_dict.get('method') == 'Hist':  # histogram method
            hist_model = True
            X_ir = self.Xhist
        else:  # naive method
            X_ir = pdIR.iloc[:, :numF_ir]
            hist_model = False

        X_all = pdIR.iloc[:, :deltaIR_max]
        y_gross = pdIR.Rev
        y_return = y_gross * nu
        Dates = pdIR.Dates

        Date_str = [a.strftime('%Y-%m-%d') for a in Dates]

        X_train, X_test, y_train, y_test, Dates_train, Dates_test = train_test_split(X_ir, y_return, Date_str,
                                                                                     test_size=0.33, random_state=seed_use)

        X_all_train, X_all_test = train_test_split(X_all, test_size=0.33, random_state=seed_use)

        mod_ridge = Ridge(alpha=alpha_use, fit_intercept=calc_b0)
        mod_ridge.fit(X_train, y_train)


        ridge_filter = mod_ridge.coef_
        filter_len = ridge_filter.shape[0]
        print "Model coefficients", ridge_filter, " b=", round(mod_ridge.intercept_, 4)
        string_write = "Model coefficients {} , b={}".format(ridge_filter, round(mod_ridge.intercept_, 4))

        y_predict = mod_ridge.predict(X_test)

        y_diff = y_test - y_predict

        l1 = y_predict.shape[0]
        y_predict_pd = np.reshape(y_predict, (l1, 1))
        y_test_pd = np.reshape(y_test, (l1, 1))
        y_diff_pd = np.reshape(y_diff, (l1, 1))
        Dates_pd = np.reshape(Dates_test, (l1, 1))

        df_dates = np.hstack((y_predict_pd, y_test_pd, y_diff_pd, Dates_pd))
        df = pd.DataFrame(df_dates,
                          columns=['PredictReturn', 'ActualReturn', 'Diff', 'Date'])

        print df.head(100)

        # now we are ready to save Ridge results
        if doSave:
            dir_path_fig = '.\Model_rez_{}'.format(
                code)
            if hist_model:
                fileName = 'Hist_IR_Ridge_{}.xlsx'.format(code)
            else:
                fileName = 'IR_Ridge_{}.xlsx'.format(code)

            if not os.path.exists(dir_path_fig):
                os.makedirs(dir_path_fig)
            path_to_fig = os.path.join(dir_path_fig, fileName)
            # df.to_csv(path_to_fig)
            writer = pd.ExcelWriter(path_to_fig)
            df.to_excel(writer, 'PredictData')
            writer.save()

            params_file = os.path.join(dir_path_fig, 'params.txt')
            file_tmp = open(params_file, 'w')
            for p in self.param_dict.items():
                file_tmp.write(str(p)+'\n')
            file_tmp.close()

            path_to_coef = os.path.join(dir_path_fig, 'coef.txt')
            # save coefficients
            file_coef = open(path_to_coef, 'w')
            file_coef.write(string_write)
            file_coef.close()

        # plot figures
        numPlot = df.shape[0]
        # numPlot = 1
        maxX = 15  # max delta to show

        for idx_plot in range(numPlot):
            cut_date = df.Date[idx_plot]
            figName = "Hist_IR_dat_code_{}_date_{}.png".format(code, cut_date)
            path_to_fig = os.path.join(dir_path_fig, figName)

            x_all = X_all_test.iloc[idx_plot, :]
            x_all = np.reshape(x_all, (deltaIR_max, 1))

            if not hist_model:
                x_predict = np.zeros(filter_len)
                for idx in range(filter_len):
                    x_predict[idx] = x_all[idx] * ridge_filter[idx]
            else:
                # x_predict = x_all[:Thr_list[-1]]
                thr0 = 0
                kIdx = 0
                x_predict = np.zeros(Thr_list[-1])
                for thr in Thr_list:
                    # x_predict[thr0:thr] = x_all[thr0:thr] * ridge_filter[kIdx]
                    for idxx in range(thr0, thr):
                        x_predict[idxx] = x_all[idxx] * ridge_filter[kIdx]
                        # print 'Thr' , thr0 , thr , 'x=' , x_predict
                    thr0 = thr
                    kIdx += 1

            x_return = x_all*nu
            time_predict = range(x_predict.shape[0])
            time_all = range(deltaIR_max)

            if True:
                fig = plt.figure(idx_plot)
                plt.plot(time_all[:maxX], x_all[:maxX], 'k-', time_all[:maxX], x_return[:maxX], 'r-', time_predict,
                         x_predict, 'b-')
                plt.legend(['All rev', 'Return rev - baseline', 'Return rev - prediction'])
                title_name = "Date={} Code={}".format(cut_date, code)
                plt.title(title_name)
                plt.xlabel('Delta days')
                plt.ylabel('Revenue')
                plt.grid()
                # figName = "Fig_code_{}_idx{}.png".format(code , plot_idx)
                if doSave:
                    fig.savefig(path_to_fig, bbox_inches='tight')
                #plt.show()
                plt.pause(0.5)






