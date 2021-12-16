# from crowdkit.datasets import load_dataset
# from crowdkit.aggregation.utils import get_accuracy
# from crowdkit.aggregation import GoldMajorityVote, MajorityVote, DawidSkene, MMSR, Wawa, ZeroBasedSkill, GLAD

import crowdkit
import load_data
import pandas as pd 
from sklearn import metrics
import click 


class NistTrecRelevance:
    '''
    List of all benchmarks:
        GoldMajorityVote, 
        MajorityVote, 
        DawidSkene, 
        MMSR, 
        Wawa, 
        ZeroBasedSkill, 
        GLAD
    '''
    
    def __init__(self, true_labels, num_labelers):
        
        self.true_labels  = true_labels
        self.num_labelers = num_labelers
        self.benchmarks   = ['GoldMajorityVote', 'MajorityVote', 'MMSR', 'Wawa', 'ZeroBasedSkill', 'GLAD', 'DawidSkene']
        
        self.crowd_labels, self.ground_truth = self.reshape_dataframe_into_this_sdk_format(true_labels)
        
    
    def apply_all_benchmarks(self):
        ''' Apply all benchmarks to the input dataset and return the accuracy and f1 score '''

        df_empty = pd.DataFrame([self.num_labelers], columns=['num_labelers']).set_index('num_labelers')
        
        
        # Measuring predicted labels for each benchmar technique:
        
        self.aggregatedLabels = df_empty.copy()
                    
        self.aggregatedLabels['GoldMajorityVote'] = crowdkit.aggregation.GoldMajorityVote().fit_predict(self.crowd_labels, self.ground_truth)        
        
        self.aggregatedLabels['MajorityVote']     = crowdkit.aggregation.MajorityVote().fit_predict(self.crowd_labels)        
        
        self.aggregatedLabels['MMSR']             = crowdkit.aggregation.MMSR(n_iter=5).fit_predict(self.crowd_labels)               
        
        self.aggregatedLabels['Wawa']             = crowdkit.aggregation.Wawa().fit_predict(self.crowd_labels)        
        
        self.aggregatedLabels['ZeroBasedSkill']   = crowdkit.aggregation.ZeroBasedSkill(n_iter=5).fit_predict(self.crowd_labels)        
        
        self.aggregatedLabels['GLAD']             = crowdkit.aggregation.GLAD(max_iter=5).fit_predict(self.crowd_labels)        
        
        self.aggregatedLabels['DawidSkene']       = crowdkit.aggregation.DawidSkene(n_iter=5).fit_predict(self.crowd_labels)



        # Measuring the Accuracy & F1-score for each benchmark:
        
        self.accuracy = df_empty.copy()
        self.f1_score = df_empty.copy()
        
        # iterate through the benchmarks 
        for benchmark in self.benchmarks:
            
            worker_label = self.aggregatedLabels[benchmark]
            
            self.accuracy[benchmark] = metrics.accuracy_score(self.ground_truth, worker_label)
            
            self.f1_score[benchmark] = metrics.f1_score(      self.ground_truth, worker_label)
                                
        
        
        return self.accuracy, self.f1_score
    
    
    def reshape_dataframe_into_this_sdk_format(self, df_true_labels):
        '''  Preprocessing the data to adapt to the sdk structure:
        '''
                
        # Converting labels from binary to integer 
        df_crowd_labels = df_true_labels.astype(int).copy()
        
        # Separating the ground truth labels from the crowd labels 
        ground_truth = df_crowd_labels.pop('truth')
            
        # Stacking all the labelers labels into one column 
        df_crowd_labels = df_crowd_labels.stack().reset_index().rename(columns={'level_0':'task', 'level_1':'performer', 0:'label'})

        # Reordering the columns to make it similar to crowd-kit examples    
        df_crowd_labels = df_crowd_labels[ ['performer','task','label'] ]
        
        return df_crowd_labels, ground_truth



@click.command()
@click.option('--dataset-name', default='ionosphere', help='Name of the dataset to be used')
def main(dataset_name = 'ionosphere'):
    
    # Loading the dataset
    data, feature_columns = load_data.aim1_3_read_download_UCI_database(WHICH_DATASET=dataset_name, mode='read')



    # generating the noisy true labels for each crowd worker
    
    ARLS = {'num_labelers': 10,  'low_dis':      0.3,   'high_dis':     0.9}

    predicted_labels, uncertainty, true_labels, labelers_strength = funcs.apply_technique_aim_1_3( data = data, ARLS = ARLS, num_simulations = 20,  feature_columns = feature_columns)



    # Finding the accuracy for all benchmark techniques
    
    NTR = NistTrecRelevance(true_labels=true_labels['train'] , num_labelers=ARLS['num_labelers'])
    
    NTR.apply_all_benchmarks()
    
    return NTR.accuracy, NTR.f1_score




if __name__ == '__main__':
    
    accuracy, f1_score = main()






''' Unnecessary functions:
    def time_gold_majority_vote(self):
        GoldMajorityVote().fit_predict(self.crowd_labels, self.ground_truth)

    def time_majority_vote(self):
        MajorityVote().fit_predict(self.crowd_labels)

    def time_dawid_skene(self):
        DawidSkene(n_iter=5).fit_predict(self.crowd_labels)

    def time_mmsr(self):
        MMSR(n_iter=5).fit_predict(self.crowd_labels)

    def time_wawa(self):
        Wawa().fit_predict(self.crowd_labels)

    def time_zbs(self):
        ZeroBasedSkill(n_iter=5).fit_predict(self.crowd_labels)

    def time_glad(self):
        GLAD(max_iter=5).fit_predict(self.crowd_labels)

    ### peak memory

    def peakmem_gold_majority_vote(self):
        GoldMajorityVote().fit_predict(self.crowd_labels, self.ground_truth)

    def peakmem_majority_vote(self):
        MajorityVote().fit_predict(self.crowd_labels)

    def peakmem_dawid_skene(self):
        DawidSkene(n_iter=5).fit_predict(self.crowd_labels)

    def peakmem_mmsr(self):
        MMSR(n_iter=5).fit_predict(self.crowd_labels)

    def peakmem_wawa(self):
        Wawa().fit_predict(self.crowd_labels)

    def peakmem_zbs(self):
        ZeroBasedSkill(n_iter=5).fit_predict(self.crowd_labels)

    def peakmem_glad(self):
        GLAD(max_iter=5).fit_predict(self.crowd_labels)

    ### accuracy    
    def _calc_accuracy(self, predict):
        predict = predict.to_frame().reset_index()
        predict.columns = ['task', 'label']
        predict['performer'] = None
        return get_accuracy(predict, true_labels=self.ground_truth)

    def track_accuracy_gold_majority_vote(self):
        return self._calc_accuracy(GoldMajorityVote().fit_predict(self.crowd_labels, self.ground_truth))

    def track_accuracy_majority_vote(self):
        return self._calc_accuracy(MajorityVote().fit_predict(self.crowd_labels))

    def track_accuracy_dawid_skene(self):
        return self._calc_accuracy(DawidSkene(n_iter=5).fit_predict(self.crowd_labels))

    def track_accuracy_mmsr(self):
        return self._calc_accuracy(MMSR(n_iter=5).fit_predict(self.crowd_labels))

    def track_accuracy_wawa(self):
        return self._calc_accuracy(Wawa().fit_predict(self.crowd_labels))

    def track_accuracy_zbs(self):
        return self._calc_accuracy(ZeroBasedSkill(n_iter=5).fit_predict(self.crowd_labels))

    def track_accuracy_glad(self):
        return self._calc_accuracy(GLAD(max_iter=5).fit_predict(self.crowd_labels))

'''