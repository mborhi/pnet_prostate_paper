import datetime
import logging
from copy import deepcopy
from os import makedirs
from os.path import join, exists
from posixpath import abspath
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from data.data_access import Data
from model.model_factory import get_model
from pipeline.one_split import OneSplitPipeline
from utils.plots import plot_box_plot
from utils.rnd import set_random_seeds

timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())


def save_model(model, model_name, directory_name):
    filename = join(abspath(directory_name), 'fs')
    logging.info('saving model {} coef to dir ({})'.format(model_name, filename))
    if not exists(filename.strip()):
        makedirs(filename)
    filename = join(filename, model_name + '.h5')
    logging.info('FS dir ({})'.format(filename))
    model.save_model(filename)


def get_mean_variance(scores):
    df = pd.DataFrame(scores)
    return df, df.mean(), df.std()


class CrossvalidationPipeline(OneSplitPipeline):
    def __init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params, exp_name):
        OneSplitPipeline.__init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params,
                                  exp_name)

    def run(self, n_splits=5):

        list_model_scores = []
        model_names = []

        for data_params in self.data_params:
            data_id = data_params['id']
            # logging
            logging.info('loading data....')
            data = Data(**data_params)

            x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data.get_train_validate_test()

            X = np.concatenate((x_train, x_validate_), axis=0)
            y = np.concatenate((y_train, y_validate_), axis=0)
            info = np.concatenate((info_train, info_validate_), axis=0)
            total_info = np.concatenate((info, info_test_), axis=0)
            total_x = np.concatenate((X, x_test_), axis=0)
            total_y = np.concatenate((y, y_test_), axis=0)
            logging.info("length of info: %s", len(total_info))

            # get model
            logging.info('fitting model ...')

            for model_param in self.model_params:
                if 'id' in model_param:
                    model_name = model_param['id']
                else:
                    model_name = model_param['type']

                set_random_seeds(random_seed=20080808)
                model_name = model_name + '_' + data_id
                m_param = deepcopy(model_param)
                m_param['id'] = model_name
                logging.info('fitting model ...')

                scores = self.train_predict_crossvalidation(m_param, X, y, info, cols, model_name, total_info, total_x, total_y)
                scores_df, scores_mean, scores_std = get_mean_variance(scores)
                list_model_scores.append(scores_df)
                model_names.append(model_name)
                self.save_score(data_params, m_param, scores_df, scores_mean, scores_std, model_name)
                logging.info('scores')
                logging.info(scores_df)
                logging.info('mean')
                logging.info(scores_mean)
                logging.info('std')
                logging.info(scores_std)

        df = pd.concat(list_model_scores, axis=1, keys=model_names)
        df.to_csv(join(self.directory, 'folds.csv'))
        plot_box_plot(df, self.directory)

        return scores_df

    def save_prediction(self, info, y_pred, y_pred_score, y_test, fold_num, model_name, training=False):
        if training:
            file_name = join(self.directory, model_name + '_traing_fold_' + str(fold_num) + '.csv')
        else:
            file_name = join(self.directory, model_name + '_testing_fold_' + str(fold_num) + '.csv')
        logging.info("saving : %s" % file_name)
        # logging.info("y_pred : %s" % len(y_pred))
        # logging.info("info : %s" % info)
        info['pred'] = y_pred
        info['pred_score'] = y_pred_score
        info['y'] = y_test
        info.to_csv(file_name)

    def custom_split_data(self, info, test_file, train_files, x, y, total_x, total_y):
        train_feature = []
        train_label = []
        train_indices = []
        all_train_infos = []
        info = pd.Series(info)
        for train_file in train_files:
            # exclude the test_file
            if test_file == train_file:
                continue
            # load the train file
            train_set = pd.read_csv(train_file)

            # validation_set = pd.read_csv(join(splits_path, 'validation_set_ILC_IDC.csv'))#'validation_set.csv'))
            # testing_set = pd.read_csv(join(splits_path, 'test_set_ILC_IDC.csv'))#'test_set.csv'))

            info_train = list(set(info).intersection(train_set.id))
            # info_validate = list(set(info).intersection(validation_set.id))
            # info_test = list(set(info).intersection(testing_set.id))

            ind_train = info.isin(info_train)
            # ind_validate = info.isin(info_validate)
            # ind_test = info.isin(info_test)

            # x_train = x[ind_train]
            x_train = total_x[ind_train]
            # x_test = x[ind_test]
            # x_validate = x[ind_validate]

            # y_train = y[ind_train]
            y_train = total_y[ind_train]
            # y_test = y[ind_test]
            # y_validate = y[ind_validate]

            info_train = info[ind_train]
            # info_test = info[ind_test]
            # info_validate = info[ind_validate]

            train_feature.append(x_train)
            train_label.append(y_train)

            # add train ind
            all_train_infos.extend(info_train)
            # train_indices.append(ind_train)
        
        # load the test file
        testing_set = pd.read_csv(test_file)

        info_test = list(set(info).intersection(testing_set.id))

        # ind_validate = info.isin(info_validate)
        ind_test = info.isin(info_test)
        
        # x_test = x[ind_test]
        x_test = total_x[ind_test]
        # x_validate = x[ind_validate]

        # y_train = y[ind_train]
        # y_test = y[ind_test]
        y_test = total_y[ind_test]
        # y_validate = y[ind_validate]

        # info_train = info[ind_train]
        info_test = info[ind_test]
        # info_validate = info[ind_validate]
        
        # return train_feature, x_test, train_label, y_test, ind_train, ind_test
        train_indices = info.isin(all_train_infos)
        return train_feature, x_test, train_label, y_test, train_indices, ind_test

    def train_predict_crossvalidation(self, model_params, X, y, info, cols, model_name, total_info, total_x, total_y):
        logging.info('model_params: {}'.format(model_params))
        n_splits = self.pipeline_params['params']['n_splits']
        skf = StratifiedKFold(n_splits=n_splits, random_state=123, shuffle=True)
        i = 0
        scores = []
        model_list = []
        """Use the splits in ./_database/prostate/splits/cross_validation_splits/"""
        # splits_path = "../../../../../../_database/prostate/splits/cross_validation_splits/" # or something
        splits_path = "../../../../../P-Net/pnet_prostate_paper/_database/prostate/splits/Pediatric_Neuroblastoma_AML/" # or something
        filepaths = [splits_path + "/" + file for file in os.listdir(splits_path)]
        for filepath in filepaths:
            x_train, x_test, y_train, y_test, train_index, test_index = self.custom_split_data(total_info, filepath, filepaths, X, y, total_x, total_y)
            model = get_model(model_params)
            logging.info('fold # ----------------%d---------' % i)
            # info_train = pd.DataFrame(index=info[train_index])
            # info_test = pd.DataFrame(index=info[test_index])
            info_train = pd.DataFrame(index=total_info[train_index])
            info_test = pd.DataFrame(index=total_info[test_index])
            x_train, x_test = self.preprocess(x_train, x_test)
            logging.info('feature extraction....')
            x_train, x_test = self.extract_features(x_train, x_test)
            flat_train_y = []
            for arr in y_train:
                flat_train_y.extend(arr)
            y_train = np.array(flat_train_y).ravel()
            # y_train = np.array(y_train)
            flat_train_x = []
            for arr in x_train:
                flat_train_x.extend(arr)
            x_train = np.array(flat_train_x)
            # logging.info('y_train: %s', np.array(y_train).ravel())
            # model = model.fit(x_train, np.array(y_train).ravel())
            # logging.info('x_train: %s', np.array(x_train).shape)
            # logging.info('y_train classes: %s', np.unique(y_train))
            model = model.fit(x_train, y_train)

            flat_test_y = []
            for arr in y_test:
                flat_test_y.extend(arr)
            y_test = np.array(flat_test_y).ravel()
            
            # flat_test_x = []
            # for arr in x_test:
            #     flat_test_x.extend(arr)
            # x_test = np.array(flat_test_x)

            y_pred_test, y_pred_test_scores = self.predict(model, x_test, y_test)
            score_test = self.evaluate(y_test, y_pred_test, y_pred_test_scores)
            logging.info('model {} -- Test score {}'.format(model_name, score_test))
            self.save_prediction(info_test, y_pred_test, y_pred_test_scores, y_test, i, model_name)

            if hasattr(model, 'save_model'):
                logging.info('saving coef')
                save_model(model, model_name + '_' + str(i), self.directory)

            if self.save_train:
                logging.info('predicting training ...')
                y_pred_train, y_pred_train_scores = self.predict(model, x_train, y_train)
                self.save_prediction(info_train, y_pred_train, y_pred_train_scores, y_train, i, model_name,
                                     training=True)

            scores.append(score_test)

            fs_parmas = deepcopy(model_params)
            if hasattr(fs_parmas, 'id'):
                fs_parmas['id'] = fs_parmas['id'] + '_fold_' + str(i)
            else:
                fs_parmas['id'] = fs_parmas['type'] + '_fold_' + str(i)

            model_list.append((model, fs_parmas))
            i += 1
        self.save_coef(model_list, cols)
        logging.info(scores)
        return scores
        """END custom splits code"""
        """
        for train_index, test_index in skf.split(X, y.ravel()):
            model = get_model(model_params)
            logging.info('fold # ----------------%d---------' % i)
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            info_train = pd.DataFrame(index=info[train_index])
            info_test = pd.DataFrame(index=info[test_index])
            x_train, x_test = self.preprocess(x_train, x_test)
            # feature extraction
            logging.info('feature extraction....')
            x_train, x_test = self.extract_features(x_train, x_test)

            model = model.fit(x_train, y_train)

            y_pred_test, y_pred_test_scores = self.predict(model, x_test, y_test)
            score_test = self.evaluate(y_test, y_pred_test, y_pred_test_scores)
            logging.info('model {} -- Test score {}'.format(model_name, score_test))
            self.save_prediction(info_test, y_pred_test, y_pred_test_scores, y_test, i, model_name)

            if hasattr(model, 'save_model'):
                logging.info('saving coef')
                save_model(model, model_name + '_' + str(i), self.directory)

            if self.save_train:
                logging.info('predicting training ...')
                y_pred_train, y_pred_train_scores = self.predict(model, x_train, y_train)
                self.save_prediction(info_train, y_pred_train, y_pred_train_scores, y_train, i, model_name,
                                     training=True)

            scores.append(score_test)

            fs_parmas = deepcopy(model_params)
            if hasattr(fs_parmas, 'id'):
                fs_parmas['id'] = fs_parmas['id'] + '_fold_' + str(i)
            else:
                fs_parmas['id'] = fs_parmas['type'] + '_fold_' + str(i)

            model_list.append((model, fs_parmas))
            i += 1
        self.save_coef(model_list, cols)
        logging.info(scores)
        return scores
        """

    def save_score(self, data_params, model_params, scores, scores_mean, scores_std, model_name):
        file_name = join(self.directory, model_name + '_params' + '.yml')
        logging.info("saving yml : %s" % file_name)
        with open(file_name, 'w') as yaml_file:
            yaml_file.write(
                yaml.dump({'data': data_params, 'models': model_params, 'pre': self.pre_params,
                           'pipeline': self.pipeline_params, 'scores': scores.to_json(),
                           'scores_mean': scores_mean.to_json(), 'scores_std': scores_std.to_json()},
                          default_flow_style=False))
