import argparse
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow.python.ops.check_ops import assert_greater_equal_v2
import load_data
from tqdm import tqdm
import numpy as np
import pandas as pd
from math import e as e_VALUE
import tensorflow.keras.backend as Keras_backend
from sklearn.ensemble import RandomForestClassifier
from scipy.special import bdtrc


def func_CallBacks(Dir_Save=''):
    mode    = 'min'
    monitor = 'val_loss'

    # checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath= Dir_Save + '/best_model_weights.h5', monitor=monitor , verbose=1, save_best_only=True, mode=mode)

    # Reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1, min_delta=0.005 , patience=10, verbose=1, save_best_only=True, mode=mode , min_lr=0.9e-5 , )

    # CSVLogger = tf.keras.callbacks.CSVLogger(Dir_Save + '/results.csv', separator=',', append=False)

    EarlyStopping = tf.keras.callbacks.EarlyStopping( monitor              = monitor, 
                                                      min_delta            = 0, 
                                                      patience             = 4, 
                                                      verbose              = 1, 
                                                      mode                 = mode, 
                                                      baseline             = 0, 
                                                      restore_best_weights = True)

    return [EarlyStopping] # [checkpointer  , EarlyStopping , CSVLogger]


def reading_terminal_inputs():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch"      , help="number of epochs")
    parser.add_argument("--bsize"      , help="batch size")
    parser.add_argument("--max_sample" , help="maximum number of training samples")
    parser.add_argument("--naug"       , help="number of augmentations")

    """ Xception          VG16                 VGG19           DenseNet201
        ResNet50          ResNet50V2           ResNet101       DenseNet169
        ResNet101V2       ResNet152            ResNet152V2     DenseNet121
        InceptionV3       InceptionResNetV2    MobileNet       MobileNetV2

        if  keras_version > 2.4
         
        EfficientNetB0     EfficientNetB1     EfficientNetB2     EfficientNetB3
        EfficientNetB4     EfficientNetB5     EfficientNetB6     EfficientNetB7 """

    parser.add_argument("--architecture_name", help='architecture name')



    args = parser.parse_args()

    epoch               = int(args.epoch)             if args.epoch             else 3
    number_augmentation = int(args.naug)              if args.naug              else 3
    bsize               = int(args.bsize)             if args.bsize             else 100
    max_sample          = int(args.max_sample)        if args.max_sample        else 1000
    architecture_name   = str(args.architecture_name) if args.architecture_name else 'DenseNet121'


    return epoch, bsize, max_sample, architecture_name, number_augmentation


def mlflow_settings():
    """
    RUN UI with postgres and HPC:
    REMOTE postgres server:
        # connecting to remote server through ssh tunneling
        ssh -L 5000:localhost:5432 artinmajdi@data7-db1.cyverse.org

        # using the mapped port and localhost to view the data
        mlflow ui --backend-store-uri postgresql://artinmajdi:1234@localhost:5000/chest_db --port 6789

    RUN directly from GitHub or show experiments/runs list:

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    
    mlflow runs list --experiment-id <id>

    mlflow run                 --no-conda --experiment-id 5 -P epoch=2 https://github.com/artinmajdi/mlflow_workflow.git -v main
    mlflow run mlflow_workflow --no-conda --experiment-id 5 -P epoch=2
    
    PostgreSQL server style
        server = f'{dialect_driver}://{username}:{password}@{ip}/{database_name}' """

    postgres_connection_type = { 'direct':     ('5432', 'data7-db1.cyverse.org'),
                                    'ssh-tunnel': ('5000', 'localhost')
                                }

    port, host = postgres_connection_type['ssh-tunnel'] # 'direct' , 'ssh-tunnel'
    username       = "artinmajdi"
    password       = '1234'
    database_name  = "chest_db_v2"
    dialect_driver = 'postgresql'
    server         = f'{dialect_driver}://{username}:{password}@{host}:{port}/{database_name}'

    Artifacts = { 'hpc':        'sftp://mohammadsmajdi@filexfer.hpc.arizona.edu:/home/u29/mohammadsmajdi/projects/mlflow/artifact_store',
                  'data7_db1':  'sftp://artinmajdi@data7-db1.cyverse.org:/home/artinmajdi/mlflow_data/artifact_store'} # :temp2_data7_b

   
    return server, Artifacts['data7_db1']


def architecture(architecture_name: str='DenseNet121', input_shape: list=[224,224,3], num_classes: int=14):
    
    input_tensor=tf.keras.layers.Input(input_shape)

    if architecture_name == 'custom':

        model = tf.keras.layers.Conv2D(4, kernel_size=(3,3), activation='relu')(input_tensor)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.MaxPooling2D(2,2)(model)
        
        model = tf.keras.layers.Conv2D(8, kernel_size=(3,3), activation='relu')(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.MaxPooling2D(2,2)(model)

        model = tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu')(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.MaxPooling2D(2,2)(model)

        model = tf.keras.layers.Flatten()(model)
        model = tf.keras.layers.Dense(32, activation='relu')(model)
        model = tf.keras.layers.Dense(num_classes , activation='softmax')(model)

        return tf.keras.models.Model(inputs=model.input, outputs=[model])

    else:

        """ Xception          VG16                 VGG19           DenseNet201
            ResNet50          ResNet50V2           ResNet101       DenseNet169
            ResNet101V2       ResNet152            ResNet152V2     DenseNet121
            InceptionV3       InceptionResNetV2    MobileNet       MobileNetV2

            if  keras_version > 2.4
            
            EfficientNetB0     EfficientNetB1     EfficientNetB2     EfficientNetB3
            EfficientNetB4     EfficientNetB5     EfficientNetB6     EfficientNetB7 """
            
        pooling='avg' 
        weights='imagenet'
        include_top=False
        
        if architecture_name == 'xception':            model_architecture = tf.keras.applications.Xception

        elif architecture_name == 'VGG16':             model_architecture = tf.keras.applications.VGG16
        elif architecture_name == 'VGG19':             model_architecture = tf.keras.applications.VGG19

        elif architecture_name == 'ResNet50':          model_architecture = tf.keras.applications.ResNet50
        elif architecture_name == 'ResNet50V2':        model_architecture = tf.keras.applications.ResNet50V2

        elif architecture_name == 'ResNet101':         model_architecture = tf.keras.applications.ResNet101
        elif architecture_name == 'ResNet101V2':       model_architecture = tf.keras.applications.ResNet101V2

        elif architecture_name == 'ResNet152':         model_architecture = tf.keras.applications.ResNet152
        elif architecture_name == 'ResNet152V2':       model_architecture = tf.keras.applications.ResNet152V2

        elif architecture_name == 'InceptionV3':       model_architecture = tf.keras.applications.InceptionV3
        elif architecture_name == 'InceptionResNetV2': model_architecture = tf.keras.applications.InceptionResNetV2

        elif architecture_name == 'MobileNet':         model_architecture = tf.keras.applications.MobileNet
        elif architecture_name == 'MobileNetV2':       model_architecture = tf.keras.applications.MobileNetV2

        elif architecture_name == 'DenseNet121':       model_architecture = tf.keras.applications.DenseNet121
        elif architecture_name == 'DenseNet169':       model_architecture = tf.keras.applications.DenseNet169
        elif architecture_name == 'DenseNet201':       model_architecture = tf.keras.applications.DenseNet201
    

        elif int(list(tf.keras.__version__)[2]) >= 4:

            if   architecture_name == 'EfficientNetB0':  model_architecture = tf.keras.applications.EfficientNetB0
            elif architecture_name == 'EfficientNetB1':  model_architecture = tf.keras.applications.EfficientNetB1
            elif architecture_name == 'EfficientNetB2':  model_architecture = tf.keras.applications.EfficientNetB2
            elif architecture_name == 'EfficientNetB3':  model_architecture = tf.keras.applications.EfficientNetB3
            elif architecture_name == 'EfficientNetB4':  model_architecture = tf.keras.applications.EfficientNetB4
            elif architecture_name == 'EfficientNetB5':  model_architecture = tf.keras.applications.EfficientNetB5
            elif architecture_name == 'EfficientNetB6':  model_architecture = tf.keras.applications.EfficientNetB6
            elif architecture_name == 'EfficientNetB7':  model_architecture = tf.keras.applications.EfficientNetB7


        model = model_architecture( weights      = weights,
                                    include_top  = include_top,
                                    input_tensor = input_tensor,
                                    input_shape  = input_shape, 
                                    pooling      = pooling) # ,classes=num_classes

        KK = tf.keras.layers.Dense( num_classes,  activation='sigmoid',  name='predictions' )(model.output)

        return tf.keras.models.Model(inputs=model.input,outputs=KK)   


def weighted_bce_loss(W):
    
    def func_loss(y_true,y_pred):

        NUM_CLASSES = y_pred.shape[1]

        loss = 0

        for d in range(NUM_CLASSES):  

            y_true = tf.cast(y_true, tf.float32)

            mask   = tf.keras.backend.cast( tf.keras.backend.not_equal(y_true[:,d], -5), 
                                            tf.keras.backend.floatx() )

            loss  += W[d]*tf.keras.losses.binary_crossentropy( y_true[:,d] * mask, 
                                                               y_pred[:,d] * mask ) 
        
        return tf.divide( loss,  tf.cast(NUM_CLASSES,tf.float32) )

    return func_loss


def optimize(dir, train_dataset, valid_dataset, epochs, Info, architecture_name):

    # architecture
    model = architecture( architecture_name = architecture_name, 
                          input_shape       = list(Info.target_size) + [3] , 
                          num_classes       = len(Info.pathologies) )
    

    model.compile( optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), 
                   loss      = weighted_bce_loss(Info.class_weights),    # tf.keras.losses.binary_crossentropy
                   metrics   = [tf.keras.metrics.binary_accuracy] )


    # optimization 
    history = model.fit( train_dataset,
                         validation_data     = valid_dataset,
                         epochs              = epochs,
                         steps_per_epoch     = Info.steps_per_epoch,
                         validation_steps    = Info.validation_steps,
                         verbose             = 1,
                         use_multiprocessing = True) #  ,callbacks=func_CallBacks(dir + '/model')

    # saving the optimized model 
    model.save( dir + '/model/model.h5', 
                overwrite         = True, 
                include_optimizer = False )
    
    return model


def evaluate(dir: str, dataset: str='chexpert', batch_size: int=1000, model=tf.keras.Model()):
    
    # Loading the data
    Data, Info = load_data.load_chest_xray( dir        = dir, 
                                            dataset    = dataset, 
                                            batch_size = batch_size, 
                                            mode       = 'test' )

    return measure_loss_acc_on_test_data(
        generator=Data.generator['test'],
        model=model,
        pathologies=Info.pathologies,
    )


def measure_loss_acc_on_test_data(generator, model, pathologies):


    # Looping over all test samples
    score_values = {}

    NUM_CLASSES = len(pathologies)
    generator.reset()
    
    for j in tqdm(range(len(generator.filenames))):

        x_test, y_test = next(generator)

        full_path, x,y = generator.filenames[j] , x_test[0,...] , y_test[0,...]
        x,y = x[np.newaxis,:] , y[np.newaxis,:]


        # Estimating the loss & accuracy for instance
        eval = model.evaluate(x=x, y=y,verbose=0,return_dict=True)

        # predicting the labels for instance
        pred = model.predict(x=x,verbose=0)

        # Measuring the loss for each class
        loss_per_class = [ tf.keras.losses.binary_crossentropy(y[...,d],pred[...,d]) for d in range(NUM_CLASSES)]

        # saving all the infos
        score_values[full_path] = {'full_path':full_path,'loss_avg':eval['loss'], 'acc_avg':eval['binary_accuracy'], 'pred':pred[0], 'pred_binary':pred[0] > 0.5, 'truth':y[0]>0.5, 'loss':np.array(loss_per_class), 'pathologies':pathologies} 


    # converting the outputs into panda dataframe
    df = pd.DataFrame.from_dict(score_values).T

    # resetting the index to integers
    df.reset_index(inplace=True)

    # # dropping the old index column
    df = df.drop(['index'],axis=1)

    return df   


class Parent_Child():
    def __init__(self, subj_info: pd.DataFrame.dtypes={}, technique: int=0, tuning_variables: dict={}):
        """ 
        
            subject_info = {'pred':[], 'loss':[], 'pathologies':['Edema','Cardiomegaly',...]}


            1. After creating a class: 
                SPC = Parent_Child(loss_dict, pred_dict, technique)

            2. Update the parent child relationship: 
            
                SPC.set_parent_child_relationship(parent_name1, child_name_list1)
                SPC.set_parent_child_relationship(parent_name2, child_name_list2)

            3. Then update the loss and probabilities

                SPC.update_loss_pred()

            4. In order to see the updated loss and probabilities use below

                loss_new_list = SPC.loss_dict_weighted  or SPC.loss_list_weighted
                pred_new_list = SPC.pred_dict_weighted  or SPC.predlist_weighted

            IMPORTANT NOTE:

                If there are more than 2 generation; it is absolutely important to enter the subjects in order of seniority 

                gen1:                grandparent (gen1)
                gen1_subjx_children: parent      (gen2)
                gen2_subjx_children: child       (gen3)

                SPC = Parent_Child(loss_dict, pred_dict, technique)

                SPC.set_parent_child_relationship(gen1_subj1, gen1_subj1_children)
                SPC.set_parent_child_relationship(gen1_subj2, gen1_subj2_children)
                                             . . .

                SPC.set_parent_child_relationship(gen2_subj1, gen2_subj1_children)
                SPC.set_parent_child_relationship(gen2_subj2, gen2_subj2_children)
                                             . . .

                SPC.update_loss_pred()
        """

        self.subj_info = subj_info        
        self.technique = technique
        self.all_parents: dict = {}
        self.tuning_variables = tuning_variables

        self.loss  = subj_info.loss
        self.pred  = subj_info.pred
        self.truth = subj_info.truth

        self._convert_inputs_list_to_dict()


    def _convert_inputs_list_to_dict(self):

        self.loss_dict  = {disease:self.subj_info.loss[index] for index,disease in enumerate(self.subj_info.pathologies)} 
        self.pred_dict  = {disease:self.subj_info.pred[index] for index,disease in enumerate(self.subj_info.pathologies)} 
        self.truth_dict = {disease:self.subj_info.truth[index] for index,disease in enumerate(self.subj_info.pathologies)} 

        self.loss_dict_weighted  = self.loss_dict
        self.pred_dict_weighted  = self.pred_dict     


    def set_parent_child_relationship(self, parent_name: str='parent_name', child_name_list: list=[]):
        self.all_parents[parent_name] = child_name_list


    def update_loss_pred(self):
        """
            techniques:
                1: coefficinet = (1 + parent_loss)
                2: coefficinet = (2 * parent_pred)
                3: coefficient = (2 * parent_pred)

                1: loss_new = loss_old * coefficient if parent_pred < 0.5 else loss_old
                2: loss_new = loss_old * coefficient if parent_pred < 0.5 else loss_old
                3. loss_new = loss_old * coefficient
        """        

        for parent_name in self.all_parents:
            self._update_loss_for_children(parent_name)

        self._convert_outputs_to_list()


    def _convert_outputs_to_list(self):
        self.loss_new = np.array([self.loss_dict_weighted[disease] for disease in self.subj_info.pathologies])
        self.pred_new = np.array([self.pred_dict_weighted[disease] for disease in self.subj_info.pathologies])


    def _update_loss_for_children(self, parent_name: str='parent_name'):

        parent_loss  = self.loss_dict_weighted[parent_name]
        parent_pred  = self.pred_dict_weighted[parent_name]
        parent_truth = self.truth_dict[parent_name]

        TV = self.tuning_variables[ self.technique ]

        if   TV['mode'] == 'truth': parent_truth_pred = parent_truth
        elif TV['mode'] == 'pred':  parent_truth_pred = parent_pred
        else:                       parent_truth_pred = 1.0


        if   self.technique == 1: coefficient = TV['weight'] * parent_loss       + TV['bias']
        elif self.technique == 2: coefficient = TV['weight'] * parent_truth_pred + TV['bias']
        elif self.technique == 3: coefficient = TV['weight'] * parent_truth_pred + TV['bias']



        for child_name in self.all_parents[parent_name]:

            new_child_loss =  self._measure_new_child_loss(coefficient, parent_name, child_name)

            self.loss_dict_weighted[child_name] = new_child_loss
            self.pred_dict_weighted[child_name] = 1 - np.power(e_VALUE , -new_child_loss)
            self.pred_dict[child_name]          = 1 - np.power(e_VALUE , -self.loss_dict[child_name])


    def _measure_new_child_loss(self, coefficient: float=0.0, parent_name: str='parent_name', child_name: str='child_name'):
        
        TV = self.tuning_variables[ self.technique ]

        parent_pred    = self.pred_dict_weighted[parent_name]
        parent_truth   = self.truth_dict[parent_name]

        if   TV['mode'] == 'truth': loss_activated = (parent_truth < 0.5 )
        elif TV['mode'] == 'pred':  loss_activated = (parent_pred  < TV['parent_pred_threshold'] )
        else:                       loss_activated = True

        old_child_loss = self.loss_dict_weighted[child_name]

        if self.technique in [1, 2]:
            if   self.technique == 1: new_child_loss = old_child_loss * coefficient if loss_activated else old_child_loss
        elif self.technique == 3:
            new_child_loss = old_child_loss * coefficient

        return new_child_loss


class Measure_InterDependent_Loss_Aim1_1(Parent_Child):
    def __init__(self,score: pd.DataFrame.dtypes={}, technique: int=0, tuning_variables: dict={}):

        score['loss_new'] = score['loss']
        score['pred_new'] = score['pred']

        self.score = score
        self.technique = technique

        for subject_ix in tqdm(self.score.index):

            Parent_Child.__init__(self, subj_info=self.score.loc[subject_ix], technique=technique, tuning_variables=tuning_variables)

            self.set_parent_child_relationship(parent_name='Lung Opacity'              , child_name_list=['Pneumonia', 'Atelectasis','Consolidation','Lung Lesion', 'Edema'])
            
            self.set_parent_child_relationship(parent_name='Enlarged Cardiomediastinum', child_name_list=['Cardiomegaly'])

            self.update_loss_pred()

            self.score.loss_new.loc[subject_ix] = self.loss_new
            self.score.pred_new.loc[subject_ix] = self.pred_new    


def apply_new_loss_techniques_aim1_1(pathologies: list=[], score: pd.DataFrame.dtypes={}, tuning_variables: dict={}):

    L = len(pathologies)

    accuracies   = np.zeros((4,L))
    measured_auc = np.zeros((4,L))

    FR = list(np.zeros(4))
    for technique in range(4):

        # extracting the ouput predictions
        if technique == 0: 
            FR[technique] = score
            output = score.pred
        else: 
            FR[technique] = Measure_InterDependent_Loss_Aim1_1(score=score, technique=technique, tuning_variables=tuning_variables)
            output = FR[technique].score.pred_new


        # Measuring accuracy
        func = lambda x1, x2: [ (x1[j] > 0.5) == (x2[j] > 0.5) for j in range(len(x1))]
        pred_acc = score.truth.combine(output,func=func).to_list()
        pred_acc = np.array(pred_acc).mean(axis=0)

        prediction_table = np.stack(score.pred)
        truth_table = np.stack(score.truth)

        for d in range(prediction_table.shape[1]):
            fpr, tpr, thresholds = roc_curve(truth_table[:,d], prediction_table[:,d], pos_label=1)
            measured_auc[technique, d] = auc(fpr, tpr)


        accuracies[technique,:] = np.floor( pred_acc*1000 ) / 10


    class Outputs:
        def __init__(self,accuracies, measured_auc, FR, pathologies):
            self.accuracy = self._converting_to_dataframe(input_table=accuracies  , columns=pathologies)
            self.auc      = self._converting_to_dataframe(input_table=measured_auc, columns=pathologies)
            self.details  = FR
            self.pathologies = pathologies

        def _converting_to_dataframe(self, input_table, columns):
            df = pd.DataFrame(input_table, columns=columns) 
            df['technique'] = ['original','1','2','3']
            df = df.set_index('technique').T

            return df
            
    return Outputs(accuracies=accuracies, measured_auc=measured_auc, FR=FR,pathologies=pathologies)

def apply_nan_back_to_truth(truth, how_to_treat_nans):

    # changing teh samples with uncertain truth label to nan
    truth[ truth == -10] = np.nan

    # how to treat the nan labels in the original dataset before measuring the average accuracy
    if   how_to_treat_nans == 'ignore': truth[ truth == -5] = np.nan
    elif how_to_treat_nans == 'pos':    truth[ truth == -5] = 1
    elif how_to_treat_nans == 'neg':    truth[ truth == -5] = 0

    return truth

def measure_mean_accruacy_chexpert(truth, prediction, how_to_treat_nans):
    """ prediction & truth: num_samples x num_classes """

    pred_classes = prediction > 0.5

    # truth_nan_applied = self._truth_with_nan_applied()
    truth_nan_applied = apply_nan_back_to_truth(truth=truth, how_to_treat_nans=how_to_treat_nans)

    # measuring the binary truth labels (the nan samples will be fixed below)
    truth_binary = truth_nan_applied > 0.5
    
    truth_pred_compare = (pred_classes == truth_binary).astype(float)

    # replacing the nan samples back to their nan value
    truth_pred_compare[np.where(np.isnan(truth_nan_applied))] = np.nan

    # measuring teh average accuracy over all samples after ignoring the nan samples
    accuracy = np.nanmean(truth_pred_compare, axis=0)*100

    # this is for safety measure; in case one of the classes overall accuracy was also nan. if removed, then the integer format below will change to very long floats
    accuracy[np.isnan(accuracy)] = 0
    accuracy = (accuracy*10).astype(int)/10
    
    return accuracy

def measure_mean_uncertainty_chexpert(truth=np.array([]), uncertainty=np.array([]), how_to_treat_nans='ignore'):

    """ uncertainty & truth:  num_samples x num_classes """

    # adding the nan values back to arrays
    truth_nan_applied = apply_nan_back_to_truth(truth, how_to_treat_nans)

    # replacing the nan samples back to their nan value
    uncertainty[np.where(np.isnan(truth_nan_applied))] = np.nan

    # measuring teh average accuracy over all samples after ignoring the nan samples
    uncertainty_mean = np.nanmean(uncertainty , axis=0)

    # this is for safety measure; in case one of the classes overall accuracy was also nan. if removed, then the integer format below will change to very long floats
    uncertainty_mean[np.isnan(uncertainty_mean)] = 0
    uncertainty_mean = (uncertainty_mean*1000).astype(int)/1000

    return uncertainty_mean
    
class Measure_Accuracy_Aim1_2():

    def __init__(self, predict_accuracy_mode: bool=False , model: tf.keras.models.Model.dtype='' , generator=tf.keras.preprocessing.image.ImageDataGenerator() , how_to_treat_nans: str='ignore', uncertainty_type: str='std'):
        """
        how_to_treat_nans:
            ignore: ignoring the nan samples when measuring the average accuracy
            pos: if integer number, it'll treat as postitive
            neg: if integer number, it'll treat as negative """

        self.predict_accuracy_mode = predict_accuracy_mode
        self.how_to_treat_nans     = how_to_treat_nans
        self.generator             = generator
        self.model                 = model 
        self.uncertainty_type      = uncertainty_type

        self._setting_params()


    def _setting_params(self):

        self.full_data_length, self.num_classes = self.generator.labels.shape

        self.batch_size     = self.generator.batch_size
        self.number_batches = int(np.ceil(self.full_data_length/self.batch_size))
        self.truth          = self.generator.labels.astype(float)

    def loop_over_whole_dataset(self):
        
        probs = np.zeros(self.generator.labels.shape)

        # Looping over all batches
        # Keras_backend.clear_session()
        self.generator.reset()
        np.random.seed(1)

        for batch_index in tqdm(range(self.number_batches),disable=False):

            # extracting the indexes for batch "batch_index"
            self.generator.batch_index = batch_index
            indexes = next(self.generator.index_generator)

            # print('   extracting data -------')                        
            self.generator.batch_index = batch_index
            x, _ = next(self.generator)

            # print('   predicting the labels -------')            
            probs[indexes,:] = self.model.predict(x,verbose=0)

        # Measuring the accuracy over whole augmented dataset
        if self.predict_accuracy_mode:
            accuracy = measure_mean_accruacy_chexpert(truth=self.truth.copy(), prediction=probs.copy(), how_to_treat_nans=self.how_to_treat_nans)
        
        return probs, accuracy

    def loop_over_all_augmentations(self,number_augmentation: int=0):

        self.number_augmentation = number_augmentation
        
        self.probs_all_augs_3d    = np.zeros((1 + number_augmentation , self.full_data_length , self.num_classes))
        self.accuracy_all_augs_3d = np.zeros((1 + number_augmentation ,                         self.num_classes))

        # Looping over all augmentation scenarios
        for ix_aug in range(number_augmentation):
            
            print(f'augmentation {ix_aug}/{number_augmentation}')
            probs, accuracy = self.loop_over_whole_dataset()

            self.probs_all_augs_3d[   ix_aug,...] = probs
            self.accuracy_all_augs_3d[ix_aug,...] = accuracy

        # measuring the average probability over all augmented data
        self.probs_avg_2d = np.mean( self.probs_all_augs_3d, axis=0)

        if self.uncertainty_type == 'std':
            self.probs_std_2d = np.std(self.probs_all_augs_3d, axis=0)


        # Measuring the accruacy for new estimated probability for each sample over all augmented data

        # self.accuracy_final    = self._measure_mean_accruacy(self.probs_avg_2d)
        # self.uncertainty_final = self._measure_mean_std(self.probs_std_2d)

        self.accuracy_final    = measure_mean_accruacy_chexpert(truth=self.truth.copy(), prediction=self.probs_avg_2d.copy(), how_to_treat_nans=self.how_to_treat_nans)
        self.uncertainty_final = measure_mean_uncertainty_chexpert(truth=self.truth.copy(), uncertainty=self.probs_std_2d.copy(), how_to_treat_nans=self.how_to_treat_nans)


def apply_technique_aim_1_2(how_to_treat_nans='ignore', data_generator='', data_generator_aug='', model='', number_augmentation=3, uncertainty_type='std'):

    print('running the evaluation on original non-augmented data')

    MA = Measure_Accuracy_Aim1_2( predict_accuracy_mode = True, 
                                  generator             = data_generator, 
                                  model                 = model, 
                                  how_to_treat_nans     = how_to_treat_nans, 
                                  uncertainty_type      = uncertainty_type)

    probs_2d_orig, old_accuracy = MA.loop_over_whole_dataset()
    



    print(' running the evaluation on augmented data including the uncertainty measurement')

    MA = Measure_Accuracy_Aim1_2( predict_accuracy_mode = True, 
                                  generator             = data_generator_aug, 
                                  model                 = model, 
                                  how_to_treat_nans     = how_to_treat_nans, 
                                  uncertainty_type      = uncertainty_type)

    MA.loop_over_all_augmentations(number_augmentation=number_augmentation)



    final_results = { 'old-accuracy': old_accuracy,
                      'new-accuracy': MA.accuracy_final, 
                      'std'         : MA.uncertainty_final}
    
    return probs_2d_orig, final_results, MA

def estimate_maximum_and_change(all_accuracies=np.array([]), pathologies=[]):
    
    columns = ['old-accuracy', 'new-accuracy', 'std']

    # creating a dataframe from accuracies
    df = pd.DataFrame(all_accuracies , index=pathologies)

    # adding the 'maximum' & 'change' columns
    df['maximum'] = df.columns[ df.values.argmax(axis=1) ]
    df['change']  = df[columns[1:]].max(axis=1) - df[columns[0]]

    # replacing "0" values to "--" for readability
    df.maximum[df.change==0.0] = '--'
    df.change[df.change==0.0] = '--'

    return df
    
# def apply_technique_aim_1_2_with_dataframe(how_to_treat_nans='ignore', pathologies=[], data_generator='', data_generator_aug='', model='', uncertainty_type='std'):
#     outputs, MA = apply_technique_aim_1_2(how_to_treat_nans=how_to_treat_nans, data_generator=data_generator, data_generator_aug=data_generator_aug, model=model, uncertainty_type=uncertainty_type)
#     df = estimate_maximum_and_change(all_accuracies=outputs, pathologies=pathologies)
#     return df, outputs, MA


""" crowdsourcing technique aim 1_3 """      

def apply_technique_aim_1_3(data={}, num_simulations=20, feature_columns=[], ARLS={}):

    def assigning_worker_true_labels(seed_num=1, true=[], labelers_strength=0.5):

        # setting the random seed
        # np.random.seed(seed_num)

        # number of samples and labelers/workers
        num_samples  = true.shape[0]

        # finding a random number for each instance
        true_label_assignment_prob = np.random.random(num_samples)

        # samples that will have an inaccurate true label
        false_samples = true_label_assignment_prob < 1 - labelers_strength

        # measuring the new labels for each labeler/worker
        worker_true = true > 0.5
        worker_true[ false_samples ] = ~ worker_true[ false_samples ] 

        return worker_true
        
    def assigning_random_labelers_strengths(num_labelers=10, low_dis=0.3, high_dis=0.9):

        labeler_names     = [f'labeler_{j}' for j in range(num_labelers)]

        # if num_labelers > 1:
        #     ls1 = np.random.uniform( low  = 0.1, 
        #                              high = 0.3, 
        #                              size = int(num_labelers/2)) 

        #     ls2 = np.random.uniform( low  = 0.7, 
        #                              high = 0.9, 
        #                              size = num_labelers - int(num_labelers/2)) 


        #     labelers_strength = np.concatenate((ls1 , ls2),axis=0)

        # else:
        labelers_strength = np.random.uniform( low  = low_dis, 
                                                high = high_dis, 
                                                size = num_labelers) 


        return pd.DataFrame( {'labelers_strength': labelers_strength}, index = labeler_names)
        

    # TODO I should repeate this for multiple seed and average
    np.random.seed(11)
    
    # setting a random strength for each labeler/worker
    labelers_strength = assigning_random_labelers_strengths( num_labelers = ARLS['num_labelers'], 
                                                             low_dis      = ARLS['low_dis'], 
                                                             high_dis     = ARLS['high_dis'])


    predicted_labels_all_sims = {'train':{},   'test':{}}
    true_labels = {'train':pd.DataFrame(),     'test':pd.DataFrame()}
    uncertainty = {'train':pd.DataFrame(),     'test':pd.DataFrame()}


    for LB_index, LB in enumerate(tqdm(labelers_strength.index, desc='workers')):
        
        # Initializationn
        for mode in ['train', 'test']:  
            predicted_labels_all_sims[mode][LB] = {}
            true_labels[mode]['truth'] = data[mode].true.copy()
    

        """ Looping over all simulations. this is to measure uncertainty """
        # extracting the simulated true labels based on the worker strength
        true_labels['train'][LB] = assigning_worker_true_labels( seed_num          = 0, # LB_index, 
                                                                 true              = data['train'].true.values, 
                                                                 labelers_strength = labelers_strength.T[LB].values )

        true_labels['test'][LB]  = assigning_worker_true_labels( seed_num          = 0, # LB_index, 
                                                                 true              = data['test'].true.values, 
                                                                 labelers_strength = labelers_strength.T[LB].values )
                                                                  
        for i in range(num_simulations):

            # training a random forest on the aformentioned labels
            RF = RandomForestClassifier( n_estimators = 5,
                                         max_depth    = 10,
                                         random_state = i)

            RF.fit( X = data['train'][feature_columns], 
                    y = true_labels['train'][LB] )


            # predicting the labels using trained networks for both train and test data
            for mode in ['train', 'test']:
                predicted_labels_all_sims[mode][LB][f'simulation_{i}'] = RF.predict( data[mode][feature_columns] )
                

        # measuring the prediction and uncertainty values after MV over all simulations
        for mode in ['train', 'test']:

            # converting to dataframe
            predicted_labels_all_sims[mode][LB] = pd.DataFrame(predicted_labels_all_sims[mode][LB], index=data[mode].index)

            # predicted probability of each class after MV over all simulations
            predicted_labels_all_sims[mode][LB]['mv'] = (predicted_labels_all_sims[mode][LB].mean(axis=1) > 0.5)

            # uncertainty for each labeler over all simulations
            uncertainty[mode][LB]                     = predicted_labels_all_sims[mode][LB].std(axis=1)


    predicted_labels = { 'train':{},  'test' :{} }
    for mode in ['train', 'test']:

        # reversing the order of simulations and labelers. NOTE: for the final experiment I should use simulation_0. if I use the mv, then because the augmented truths keeps changing in each simulation, then with enough simulations, I'll end up witht perfect labelers.
        for i in range(num_simulations + 1):
            SM = f'simulation_{i}' if i < num_simulations else 'mv'

            predicted_labels[mode][SM] = pd.DataFrame()
            for LB in [f'labeler_{j}' for j in range(ARLS['num_labelers'])]:
                predicted_labels[mode][SM][LB]  = predicted_labels_all_sims[mode][LB][SM]
                

    labelers_strength['accuracy-test'] = 0
    acc = {}
    for i in range(ARLS['num_labelers']):
        LB = f'labeler_{i}'
        labelers_strength.loc[LB,'accuracy-test'] = ( predicted_labels['test']['mv'][LB] == true_labels['test'].truth ).mean()


    return predicted_labels, uncertainty, true_labels, labelers_strength


def aim1_3_measuring_weights(labels_all_workers, uncertainty_all_workers):

    # weights       : num_labelers x num_methods
    # prob_weighted : num_samples x num_labelers

    prob_mv_binary = labels_all_workers.mean(axis=1) > 0.5

    T1, T2, w_hat1, w_hat2 = {}, {}, {}, {}

    for workers_name in labels_all_workers.columns:

        T1[workers_name] = 1 - uncertainty_all_workers[workers_name]

        T2[workers_name] = T1[workers_name].copy()
        T2[workers_name][ labels_all_workers[workers_name].values != prob_mv_binary.values ] = 0

        w_hat1[workers_name] = T1[workers_name].mean(axis=0)
        w_hat2[workers_name] = T2[workers_name].mean(axis=0)


    w_hat = pd.DataFrame([w_hat1, w_hat2], index=['method1', 'method2']).T

    # measuring average weight
    weights = w_hat.divide(w_hat.sum(axis=0),axis=1)

    prob_weighted = pd.DataFrame()
    for method in ['method1', 'method2']:
        prob_weighted[method] =( labels_all_workers * weights[method] ).sum(axis=1)

    return weights, prob_weighted


def aim1_3_measuring_benchmark_accuracy(delta , noisy_true_labels):
    """
        tau          : 1 x 1
        weights_Tao  : num_samples x num_labelers
        W_hat_Tao    : num_samples x num_labelers 
        z            : num_samples x 1
        gamma        : num_samples x 1
    """


    tau =  ( delta == noisy_true_labels ).mean(axis=0)

    # number of labelers
    M = len(delta.columns)

    # number of true and false labels for each class and sample
    true_counts  = delta.sum(axis=1)
    false_counts = M - true_counts

    # measuring the "specific quality of instanses"
    s           = delta.multiply(true_counts-1,axis=0) + (~delta).multiply(false_counts-1,axis=0)
    gamma       = (1 + s ** 2) * tau
    W_hat_Tao   = gamma.applymap(lambda x: 1/(1 + np.exp(-x)) )
    z           = W_hat_Tao.mean(axis=1)
    return W_hat_Tao.divide(z , axis=0)



def aim1_3_measure_confidense_score(delta, weights, conf_strategy, num_labelers, truth):

    def measuring_accuracy(positive_probs, truth):
        """ Measuring accuracy. This result in the same values as if I had measured a weighted majorith voting using the "weights" multiplied by "delta" which is the binary predicted labels """
        
        return ( (positive_probs > 0.5) == truth ).mean(axis=0)


    P_pos = (  delta * weights ).sum(axis=1)
    P_neg = ( ~delta * weights ).sum(axis=1)
        
    if conf_strategy in (1,'freq'):
        
        F = P_pos.copy()
        F[P_pos < P_neg] = P_neg[P_pos < P_neg]

        Accuracy = measuring_accuracy(positive_probs=P_pos, truth=truth)
        
    elif conf_strategy in (2,'beta'):

        # f_pos = pd.DataFrame( np.zeros((delta.shape[0],2)) , index=delta.index )
        # f_neg = pd.DataFrame( np.zeros((delta.shape[0],2)) , index=delta.index )

        # for method in ['method1', 'method2']:
        f_pos = 1 + P_pos * num_labelers # num_labelers * (  delta * weights ).sum(axis=1)
        f_neg = 1 + P_neg * num_labelers # num_labelers * ( ~delta * weights ).sum(axis=1)



        k_df = f_neg.floordiv(1)
        n_df = (f_neg + f_pos).floordiv(1) - 1

        I = k_df.copy()
        # for method in ['method1','method2']:
        for index in n_df.index:

            # k = k_df[index]
            # n = n_df[index]
            # p = 0.5

            I[index] = bdtrc( k_df[index] , n_df[index] , 0.5 )

        # I.hist()

        F = I.copy()
        F[I < 0.5] = (1-I)[I < 0.5]
        # F.hist()

        Accuracy = measuring_accuracy(positive_probs=I, truth=truth)

    
    return F, Accuracy



def aim1_3_full_accuracy_comparison(data=[], num_labelers=10, num_simulations=20, feature_columns=[]):

    ARLS = {'num_labelers': num_labelers, 
            'low_dis':      0.3, 
            'high_dis':     0.9}
        
    predicted_labels, uncertainty, true_labels, labelers_strength = apply_technique_aim_1_3( data = data,
                                                                                                ARLS = ARLS,
                                                                                                num_simulations = num_simulations, 
                                                                                                feature_columns = feature_columns)

    labels_all_workers         = predicted_labels['test']['mv'] 
    uncertainty_all_workers    = uncertainty['test']
    truth                      = true_labels['test'].truth


    """ 
    Measuring the new accuracies - Confidence score
        - weights dimentions:            (num_labelers x num_methods)
        - labels_all_workers (Binary):   (num_samples  x num_labelers)
        - prob_weighted dimentions:      (num_samples  x num_methods)
        - true_labels dimentions:        (num_samples  x 1) 
    """


    # Measuring weights for the proposed technique
    weights_proposed, prob_weighted = aim1_3_measuring_weights( labels_all_workers      = labels_all_workers, 
                                                                uncertainty_all_workers = uncertainty_all_workers)
                                                                

    # Benchmark accuracy measurement
    weights_Tao = aim1_3_measuring_benchmark_accuracy( delta             = predicted_labels['test']['simulation_0'], 
                                                       noisy_true_labels = true_labels['test'].drop(columns=['truth']) )
                                                    

    F, accuracy = {} , {}
    for strategy in ['freq', 'beta']:


        F[strategy], accuracy[strategy]  = pd.DataFrame() , pd.DataFrame(index=[num_labelers])

        for method in ['method1', 'method2', 'Tao', 'Sheng']: # Tao: wMV-freq  Sheng: MV-freq

            if   method in ['method1', 'method2']: weights = weights_proposed[method]
            elif method in ['Tao']:                weights = weights_Tao / num_labelers
            elif method in ['Sheng']: weights = pd.DataFrame( 1 / num_labelers , index=weights_Tao.index, columns=weights_Tao.columns)


            F[strategy][method], accuracy[strategy][method] = aim1_3_measure_confidense_score(  delta         = labels_all_workers,
                                                                                                weights       = weights,
                                                                                                conf_strategy = strategy,
                                                                                                num_labelers  = num_labelers,
                                                                                                truth         = truth )


        accuracy[strategy]['MV'] = ( (labels_all_workers.sum(axis=1) > 0.5) == truth ).mean(axis=0)

        # F_freq,        F_beta        = F['freq'],        F['beta']
        # accuracy_freq, accuracy_beta = accuracy['freq'], accuracy['beta']
    return F, accuracy