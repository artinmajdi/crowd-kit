from logging import error, warning
import pandas as pd
import numpy as np
import tensorflow as tf 
import os 
import wget 
    
def nih(dir: str, max_sample: int):
    
    """ reading the csv tables """    
    all_data       = pd.read_csv(dir + '/files/Data_Entry_2017_v2020.csv')
    test_list      = pd.read_csv(dir + '/files/test_list.txt', names=['Image Index'])
      
    
    
    """ Writing the relative path """     
    all_data['Path']      = 'data/' + all_data['Image Index']
    all_data['full_path'] = dir +'/data/' + all_data['Image Index']
    
    
    
    """ Finding the list of all studied pathologies """    
    all_data['Finding Labels'] = all_data['Finding Labels'].map(lambda x: x.split('|'))
    # pathologies = set(list(chain(*all_data['Finding Labels'])))
   

    
    """ overwriting the order of pathologeis """    
    pathologies = ['No Finding', 'Pneumonia', 'Mass', 'Pneumothorax', 'Pleural_Thickening', 'Edema', 'Cardiomegaly', 'Emphysema', 'Effusion', 'Consolidation', 'Nodule', 'Infiltration', 'Atelectasis', 'Fibrosis']



    """ Creating the pathology based columns """    
    for name in pathologies:
        all_data[name] = all_data['Finding Labels'].map(lambda x: 1 if name in x else 0)

        

    """ Creating the disease vectors """        
    all_data['disease_vector'] = all_data[pathologies].values.tolist()
    all_data['disease_vector'] = all_data['disease_vector'].map(lambda x: np.array(x))  
   
    
   
    """ Selecting a few cases """    
    all_data = all_data.iloc[:max_sample,:]
    
    
    
    """ Removing unnecessary columns """    
    # all_data = all_data.drop(columns=['OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x',	'y]', 'Follow-up #'])


    
    """ Delecting the pathologies with at least a minimum number of samples """    
    # MIN_CASES = 1000
    # pathologies = [name for name in pathologies if all_data[name].sum()>MIN_CASES]
    # print('Number of samples per class ({})'.format(len(pathologies)), 
    #     [(name,int(all_data[name].sum())) for name in pathologies])
       
        
    
    """ Resampling the dataset to make class occurrences more reasonable """
    # CASE_NUMBERS = 800
    # sample_weights = all_data['Finding Labels'].map(lambda x: len(x) if len(x)>0 else 0).values + 4e-2
    # sample_weights /= sample_weights.sum()
    # all_data = all_data.sample(CASE_NUMBERS, weights=sample_weights)

    
    
    """ Separating train validation test """    
    test      = all_data[all_data['Image Index'].isin(test_list['Image Index'])]
    train_val = all_data.drop(test.index)

    valid     = train_val.sample(frac=0.2,random_state=1)
    train     = train_val.drop(valid.index)

    print('after sample-pruning')
    print('train size:',train.shape)
    print('valid size:',valid.shape)
    print('test size:' ,test.shape) 
    
    
    
    """ Class weights """
    L = len(pathologies)
    class_weights = np.ones(L)/L
    
    
    
    return train, valid, test, pathologies, class_weights


def chexpert(dir: str, max_sample: int):
    
    """ Selecting the pathologies """    
    pathologies = ["No Finding", "Enlarged Cardiomediastinum" , "Cardiomegaly" , "Lung Opacity" , "Lung Lesion", "Edema" , "Consolidation" , "Pneumonia" , "Atelectasis" , "Pneumothorax" , "Pleural Effusion" , "Pleural Other" , "Fracture" , "Support Devices"]
    
    
    """ Loading the raw table """
    # train = pd.read_csv(dir + '/train_aim1_2.csv')
    train = pd.read_csv(dir + '/train.csv')
    test  = pd.read_csv(dir + '/valid.csv')

    print('before sample-pruning')
    print('train:',train.shape)
    print('test:',test.shape)
    
    """ Label Structure
        positive (exist):            1.0
        negative (doesn't exist):   -1.0
        Ucertain                     0.0
        no mention                   nan """
    
    """ Adding full directory """    
    train['full_path'] = dir +'/' + train['Path']
    test['full_path'] = dir +'/' + test['Path']
    
    
    
    """ Extracting the pathologies of interest """    
    train = cleaning_up_dataframe(train, pathologies, 'train')
    test  = cleaning_up_dataframe(test, pathologies , 'test')


    """ Selecting a few cases """    
    train = train.iloc[:max_sample,:]
    test  = test.iloc[:max_sample ,:]


    """ Separating the uncertain samples """    
    train_uncertain = train.copy()
    for name in pathologies:
        train = train.loc[train[name]!='uncertain']
        
    train_uncertain = train_uncertain.drop(train.index)


    """ Splitting train/validatiion """    
    valid = train.sample(frac=0.2,random_state=1)
    train = train.drop(valid.index)
    
    
    print('\nafter sample-pruning')
    print('train (certain):',train.shape)
    print('train (uncertain):',train_uncertain.shape)
    print('valid:',valid.shape)
    print('test:',test.shape,'\n')
    
    
    # TODO make no finding 0 for all samples where we at least have one case
    """ Changing classes from string to integer 
        Tagging the missing labels; this number "-0.5" will later be masked during measuring the loss """   

    train_uncertain = train_uncertain.replace('pos',1).replace('neg',0).replace(np.nan,-5.0).replace('uncertain',-10.0)
    train = train.replace('pos',1).replace('neg',0).replace(np.nan,-5.0)
    valid = valid.replace('pos',1).replace('neg',0).replace(np.nan,-5.0)
    test  = test.replace('pos',1).replace('neg',0)
    
    
    """ Changing the nan values for parents with at lease 1 TRUE child to TRUE """
    train_uncertain = replacing_parent_nan_values_with_one_if_child_exist(train_uncertain)
    train = replacing_parent_nan_values_with_one_if_child_exist(train)
    valid = replacing_parent_nan_values_with_one_if_child_exist(valid)



    """ Class weights """    
    L = len(pathologies)
    class_weights = np.ones(L)/L
    
    return (train, train_uncertain), valid, test, pathologies, class_weights


def cleaning_up_dataframe(data, pathologies, mode):
    """ Label Structure
        positive (exist):            1.0
        negative (doesn't exist):   -1.0
        Ucertain                     0.0
        no mention                   nan """

    # changing all no mention labels to negative
    data = data[data['AP/PA']=='AP']
    data = data[data['Frontal/Lateral']=='Frontal']


    # Treat all other nan s as negative
    # data = data.replace(np.nan,-1.0)


    # renaming the pathologeis to 'neg' 'pos' 'uncertain'
    for column in pathologies:
        
        data[column] = data[column].replace(1,'pos')
        
        if mode == 'train':
            data[column] = data[column].replace(-1,'neg')
            data[column] = data[column].replace(0,'uncertain')
        elif mode == 'test':
            data[column] = data[column].replace(0,'neg')
            

    # according to CheXpert paper, we can assume all pathologise are negative when no finding label is True
    no_finding_indexes = data[data['No Finding']=='pos'].index
    for disease in pathologies:
        if disease != 'No Finding':
            data.loc[no_finding_indexes, disease] = 'neg'


    return data


def replacing_parent_nan_values_with_one_if_child_exist(data: pd.DataFrame):

    """     parent ->
                - child
 
            Lung Opacity -> 

                - Pneuomnia
                - Atelectasis
                - Edema
                - Consolidation
                - Lung Lesion

            Enlarged Cardiomediastinum -> 

                - Cardiomegaly       """


    func = lambda x1, x2: 1.0 if np.isnan(x1) and x2==1.0 else x1

    for child_name in ['Pneumonia','Atelectasis','Edema','Consolidation','Lung Lesion']:

        data['Lung Opacity'] = data['Lung Opacity'].combine(data[child_name], func=func)


    for child_name in ['Cardiomegaly']:

        data['Enlarged Cardiomediastinum'] = data['Enlarged Cardiomediastinum'].combine(data[child_name], func=func)

    return data


def load_chest_xray(dir='', dataset='chexpert', batch_size=30, mode='train_val', max_sample=100000):

    # output_shapes = ([None,224,224,3],[None,len(pathologies)]) 
    # output_types  = (tf.float32,tf.float32)
    # target_size   =  (224,224)

    class Info_Class:
        def __init__(self, pathologies: list=[], class_weights: list=[], target_size: tuple=(224,224), steps_per_epoch: int=0, validation_steps: int=0):
            self.pathologies      = pathologies
            self.class_weights    = class_weights
            self.target_size      = target_size
            self.steps_per_epoch  = steps_per_epoch
            self.validation_steps = validation_steps


    class Data_Class:
        def __init__(self, data_tf, generator, dataframe):
            self.data_tf   = data_tf
            self.dataframe = dataframe
            self.generator = generator


    def create_generator(dataframe: pd.DataFrame.dtypes, augmentaion: bool=False, target_size: tuple=(224,224)):

        """           
            zca_epsilon 	                epsilon for ZCA whitening. Default is 1e-6.
            zca_whitening 	                Boolean. Apply ZCA whitening.
            rotation_range 	                Int. Degree range for random rotations. 
            
            featurewise_center 	            Boolean. Set input mean to 0 over the dataset, feature-wise.
            samplewise_center 	            Boolean. Set each sample mean to 0.
            featurewise_std_normalization 	Boolean. Divide inputs by std of the dataset, feature-wise.
            samplewise_std_normalization 	Boolean. Divide each input by its std.

            height_shift_range
                float: fraction of total width, if < 1, or pixels if >= 1.
                1-D array-like: random elements from the array.
                int: integer number of pixels from interval (-width_shift_range, +width_shift_range)

                e.g., With width_shift_range=2 possible values are integers [-1, 0, +1], same as with width_shift_range=[-1, 0, +1], while with width_shift_range=1.0 possible values are floats in the interval [-1.0, +1.0). 

            width_shift_range:  (similar to above)

            shear_range:
                Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)

            zoom_range:       
                Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] =     [1-zoom_range, 1+zoom_range]. 

            fill_mode:
                One of {"constant", "nearest", "reflect" or "wrap"}. Default is 'nearest'. Points outside the boundaries of the input are filled according to the given mode: 

            data_format:
                Image data format, either "channels_first" or "channels_last". "channels_last" mode means that the images should have shape (samples, height, width, channels), "channels_first" mode means that the images should have shape (samples, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last". 

            validation_split     
                Float. Fraction of images reserved for validation (strictly between 0 and 1). 
            
            dtype                
                Dtype to use for the generated arrays. 

            rescale:
                rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (after applying all other transformations). 

            preprocessing_function:
                function that will be applied on each input. The function will run after the image is resized and augmented. The function should take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape. 
        """

        # the true color_mode is grayscale. but because densenet input is rgb, is set to rgb
        color_mode = 'rgb' 
        y_col      = pathologies
        class_mode = 'raw'
        
        # Creating the generator
        if not augmentaion:
            generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        else:
            generator = tf.keras.preprocessing.image.ImageDataGenerator(
                fill_mode           = 'nearest',
                rescale             = 1./255, 
                rotation_range      = 15, 
                height_shift_range  = 0.1,
                width_shift_range   = 0.1,
                # zoom_range          = 0.1,
                # shear_range         = 10,
                
                horizontal_flip     = False, 
                vertical_flip       = False,
                featurewise_center  = False, 
                samplewise_center   = False,
                featurewise_std_normalization = False, 
                samplewise_std_normalization  = False,
                )


        # Loading the data from physical storage
        data_generator = generator.flow_from_dataframe(
            dataframe   = dataframe, 
            x_col       = 'Path', 
            y_col       = y_col,
            color_mode  = color_mode,
            directory   = dir, 
            target_size = target_size, 
            batch_size  = batch_size, 
            class_mode  = class_mode, 
            shuffle     = False, 
            classes     = y_col,
            )

        steps_per_epoch = int(len(data_generator.filenames)/batch_size)

        return data_generator, steps_per_epoch


    # Loading the pre-processed dataframe
    if dataset == 'nih':
        df_train, df_valid, df_test, pathologies, class_weights = nih(dir,max_sample)

    elif dataset == 'chexpert':        
        (df_train, df_train_uncertain), df_valid, df_test, pathologies, class_weights = chexpert(dir,max_sample)
        
    
    # Keras Generator
    output_shapes = ([None,224,224,3],[None,len(pathologies)]) 
    output_types  = (tf.float32,tf.float32)
    target_size   =  (224,224)



    if mode in ('train_val', 'valid', 'valid_df'):
        
        if mode in ('train_val'):
            # creating the data generator for train data
            generator_train, steps_per_epoch = create_generator(dataframe=df_train, augmentaion=False, target_size=target_size)
            data_train  = tf.data.Dataset.from_generator(lambda: generator_train,output_types=output_types,output_shapes=output_shapes)

        elif mode in ('valid', 'valid_df'):
            generator_train, steps_per_epoch, data_train= '', '', ''


        # creating the data generator for valid data
        if mode in ('train_val', 'valid'):
            generator_valid, validation_steps  = create_generator(dataframe=df_valid, augmentaion=False, target_size=target_size)
            data_valid = tf.data.Dataset.from_generator(lambda: generator_valid,output_types=output_types,output_shapes=output_shapes)
            generator_valid_aug, _ = create_generator(dataframe=df_valid, augmentaion=True , target_size=target_size)

        elif mode == 'valid_df':
            generator_valid, validation_steps, data_valid, generator_valid_aug= '', '', '', ''

        # creating the info class
        Info = Info_Class(pathologies      = pathologies, 
                          class_weights    = class_weights, 
                          target_size      = target_size, 
                          validation_steps = validation_steps, 
                          steps_per_epoch  = steps_per_epoch)

        Data = Data_Class(data_tf   = {'train':data_train,      'valid':data_valid},
                          generator = {'train':generator_train, 'valid':generator_valid, 'valid_aug':generator_valid_aug},
                          dataframe = {'train':df_train,        'valid':df_valid,        'uncertain':df_train_uncertain, 'test':df_test})

        return Data, Info



    elif mode == 'test':
        
        # creating the data generator for uncertain data with & without augmentation
        generator_test, validation_steps = create_generator(dataframe=df_test, augmentaion=False, target_size=target_size)
        generator_test_aug, _ = create_generator(dataframe=df_test, augmentaion=True , target_size=target_size)

        # creating the info class
        Info = Info_Class(pathologies      = pathologies, 
                          class_weights    = class_weights, 
                          target_size      = target_size, 
                          validation_steps = validation_steps, 
                          steps_per_epoch  = '')

        Data = Data_Class(data_tf   = {},
                          generator = {'test':generator_test, 'test_aug':generator_test_aug},
                          dataframe = {'train':df_train, 'valid':df_valid, 'uncertain':df_train_uncertain, 'test':df_test})

        return Data, Info



    elif mode == 'uncertain':

        # creating the data generator for uncertain data with & without augmentation
        generator_uncertain, validation_steps = create_generator(dataframe=df_train_uncertain, augmentaion=False, target_size=target_size)
        generator_uncertain_aug, _            = create_generator(dataframe=df_train_uncertain, augmentaion=True , target_size=target_size)

        # creating the info class
        Info = Info_Class(pathologies      = pathologies, 
                          class_weights    = class_weights, 
                          target_size      = target_size, 
                          validation_steps = validation_steps, 
                          steps_per_epoch  = '')

        Data = Data_Class(data_tf   = {},
                          generator = {'uncertain':generator_uncertain, 'uncertain_aug':generator_uncertain_aug},
                          dataframe = {'train':df_train, 'valid':df_valid, 'uncertain':df_train_uncertain, 'test':df_test})

        return Data, Info


def load_chest_xray_with_mode(dataset: str='chexpert', mode: str='train_val', max_sample: int=1000000):
    """ this function is to just functionalize this specific usage in multiple notebooks and scripts """

    dir = '/groups/jjrodrig/projects/chest/dataset/' + dataset + '/'

    # loading the data
    if mode in ('train_val',  'valid'):
        Data, Info = load_chest_xray(dir=dir, dataset=dataset, batch_size=100, mode=mode, max_sample=max_sample)    
        data_generator     = Data.generator['valid']
        data_generator_aug = Data.generator['valid_aug']

    elif mode == 'uncertain':
        Data, Info = load_chest_xray(dir=dir, dataset=dataset, batch_size=100, mode=mode, max_sample=max_sample)
        data_generator     = Data.generator['uncertain']
        data_generator_aug = Data.generator['uncertain_aug']

    elif mode == 'test':
        Data, Info = load_chest_xray(dir=dir, dataset=dataset, batch_size=100, mode=mode, max_sample=max_sample)
        data_generator     = Data.generator['test']
        data_generator_aug = Data.generator['test_aug']

    return Data, Info, data_generator, data_generator_aug

    
def aim1_3_read_download_UCI_database(WHICH_DATASET=5, mode='read'):

    # main directory
    local_parent_path = os.path.dirname(os.path.dirname(__file__)) + '/data_mine'
    
    if not os.path.isdir(local_parent_path):
        print('directory not found', local_parent_path)
    else:
        print('directory found', local_parent_path)
        
    def read_raw_names_files(WHICH_DATASET=1):

        main_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


        if WHICH_DATASET in (1,'kr-vs-kp'):
            dataset = 'kr-vs-kp'
            names   = [f'a{i}' for i in range(0,36)] + ['true']
            files   = ['Index', f'{dataset}.data', f'{dataset}.names']
            url     = main_url + '/chess/king-rook-vs-king-pawn/'

        elif WHICH_DATASET in (2,'mushroom'):
            dataset = 'agaricus-lepiota'
            names   = ['true'] + [f'a{i}' for i in range(0,22)]
            files   = ['Index', f'{dataset}.data', f'{dataset}.names']
            url     = main_url + '/mushroom/'

        elif WHICH_DATASET in (3,'sick'):
            dataset = 'sick'
            names   = [f'a{i}' for i in range(0,29)] + ['true']
            files   = [f'{dataset}.data', f'{dataset}.names', f'{dataset}.test']
            url     = main_url + '/thyroid-disease/'

        elif WHICH_DATASET in (4,'spambase'):
            dataset = 'spambase'
            names   = [f'a{i}' for i in range(0,57)] + ['true']
            files   = [f'{dataset}.DOCUMENTATION', f'{dataset}.data', f'{dataset}.names', f'{dataset}.zip']
            url     = main_url + '/spambase/'
            
        elif WHICH_DATASET in (5,'tic-tac-toe'):
            dataset = 'tic-tac-toe'
            names   = [f'a{i}' for i in range(0,9)] + ['true']
            files   = [f'{dataset}.data', f'{dataset}.names']
            url     = main_url + '/tic-tac-toe/'

        elif WHICH_DATASET in (6, 'splice'):
            # dataset = 'splice'
            # url = main_url + '/molecular-biology/splice-junction-gene-sequences/'
            pass

        elif WHICH_DATASET in (7,'thyroid'):
            pass
        
        elif WHICH_DATASET in (8,'waveform'):
            dataset = 'waveform'
            names   = [f'a{i}' for i in range(0,21)] + ['true']
            files   = [ 'Index', f'{dataset}-+noise.c', f'{dataset}-+noise.data.Z', f'{dataset}-+noise.names', f'{dataset}.c', f'{dataset}.data.Z', f'{dataset}.names']
            url     = main_url + '/mwaveform/'

        elif WHICH_DATASET in (9,'biodeg'):
            dataset = 'biodeg'
            names   = [f'a{i}' for i in range(0,41)] + ['true']
            files   = [f'{dataset}.csv']
            url     = main_url + '/00254/'

        elif WHICH_DATASET in (10,'horse-colic'):
            dataset = 'horse-colic'
            names   = [f'a{i}' for i in range(0,41)] + ['true']
            files   = [f'{dataset}.data', f'{dataset}.names', f'{dataset}.names.original', f'{dataset}.test']
            url     = main_url + '/horse-colic/'
            
        elif WHICH_DATASET in (11,'ionosphere'):
            dataset = 'ionosphere'
            names   = [f'a{i}' for i in range(0,34)] + ['true']
            files   = [ 'Index', f'{dataset}.data', f'{dataset}.names']
            url     = main_url + '/ionosphere/'

        elif WHICH_DATASET in (12,'vote'):
            pass  

        return dataset, names, files, url

    def download_data(local_parent_path=''):

        dataset, _, files, url = read_raw_names_files(WHICH_DATASET=WHICH_DATASET)

        local_path = f'{local_parent_path}/UCI_{dataset}'

        if not os.path.isdir(local_path):  
            os.mkdir(local_path) 
        
        for name in files: 
            wget.download(url + name, local_path)

        data_raw = pd.read_csv( local_parent_path + f'/UCI_{dataset}/{dataset}.data')


        return data_raw, []

    def read_data(local_parent_path='', WHICH_DATASET=0):

        def postprocess(data_raw=[], names=[], WHICH_DATASET=0):

            def replacing_classes_char_to_int(data_raw=[], feature_columns=[]):
                
                # finding the unique classes
                lbls = set()
                for fx in feature_columns:
                    lbls = lbls.union(data_raw[fx].unique())

                # replacing the classes from char to int
                for ix, lb in enumerate(lbls):
                    data_raw[feature_columns] = data_raw[feature_columns].replace(lb,ix+1)

                return data_raw
                
            feature_columns = names.copy()
            feature_columns.remove('true')

            if WHICH_DATASET in (1,'kr-vs-kp'):

                # changing the true labels from string to [0,1]
                data_raw.true = data_raw.true.replace('won',1).replace('nowin',0)

                # replacing the classes from char to int
                data_raw = replacing_classes_char_to_int(data_raw, feature_columns)

            elif WHICH_DATASET in (2,'mushroom'):
                
                # changing the true labels from string to [0,1]
                data_raw.true = data_raw.true.replace('e',1).replace('p',0)

                # feature a10 has missing data
                data_raw.drop(columns=['a10'], inplace=True)
                feature_columns.remove('a10')

                # replacing the classes from char to int
                data_raw = replacing_classes_char_to_int(data_raw, feature_columns)

            elif WHICH_DATASET in (3,'sick'):
                data_raw.true = data_raw.true.map(lambda x: x.split('.')[0]).replace('sick',1).replace('negative',0)
                data_raw = data_raw.replace('?',np.nan).drop(columns=['a27'])

            elif WHICH_DATASET in (4,'spambase'): 
                pass

            elif WHICH_DATASET in (5,'tic-tac-toe'):
                # renaming the two classes "good" and "bad" to "0" and "1"
                data_raw.true = data_raw.true.replace('negative',0).replace('positive',1)
                data_raw[feature_columns] = data_raw[feature_columns].replace('x',1).replace('o',2).replace('b',0)

            elif WHICH_DATASET in (6, 'splice'):
                pass 

            elif WHICH_DATASET in (7,'thyroid'):
                pass  

            elif WHICH_DATASET in (8,'waveform'):
                # extracting only classes "1" and "2" to correspond to Tao et al paper
                class_0 = data_raw[data_raw.true == 0].index
                data_raw.drop(class_0, inplace=True)
                data_raw.true = data_raw.true.replace(1,0).replace(2,1)

            elif WHICH_DATASET in (9,'biodeg'):
                data_raw.true = data_raw.true.replace('RB',1).replace('NRB',0)

            elif WHICH_DATASET in (10,'horse-colic'): 
                pass

            elif WHICH_DATASET in (11,'ionosphere'):
                data_raw.true = data_raw.true.replace('g',1).replace('b',0)

            elif WHICH_DATASET in (12,'vote'):
                pass

            return data_raw, feature_columns

        def separate_train_test(data_raw=[], train_frac=0.8):
            data = {}
            data['train'] = data_raw.sample(frac=train_frac).sort_index()
            data['test']  = data_raw.drop(data['train'].index)
            
            return data

            
        dataset, names, _, _ = read_raw_names_files(WHICH_DATASET=WHICH_DATASET)


        if dataset == 'biodeg':        
            command = {'filepath_or_buffer': local_parent_path + f'/UCI_{dataset}/{dataset}.csv', 'delimiter':';'}

        elif dataset == 'horse-colic': 
            command = {'filepath_or_buffer': local_parent_path + f'/UCI_{dataset}/{dataset}.data', 'delimiter':' ', 'index_col':None}

        else:                   
            command = {'filepath_or_buffer': local_parent_path + f'/UCI_{dataset}/{dataset}.data'}
                            
        if mode == 'read':
            data_raw = pd.read_csv(**command, names=names)
            data_raw, feature_columns = postprocess(data_raw=data_raw, names=names, WHICH_DATASET=WHICH_DATASET)

        elif mode == 'read_raw':
            data_raw, feature_columns = pd.read_csv(**command) , []

        data = separate_train_test(data_raw=data_raw, train_frac=0.8)

        return data, feature_columns


    if   'download' in mode: 
        return download_data(local_parent_path=local_parent_path)
        
    elif 'read'     in mode: 
        return read_data(    local_parent_path=local_parent_path, WHICH_DATASET=WHICH_DATASET)




if __name__ == '__main__':

    dataset = 'chexpert'
    dir = '/groups/jjrodrig/projects/chest/dataset/' + dataset + '/'
    chexpert(dir=dir, max_sample=1000)
    