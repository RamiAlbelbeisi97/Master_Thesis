# for producing random samples
import random
# for plots
import matplotlib.pyplot as plt
# for math
import numpy as np
#for visualizing high dimensional inputs
import umap
# fast file loading/saving
import pickle
# file accessing
import glob
# file searching
import re
#reading simulation outputs
import yt
#timing outputs
import time
#file finding package
import os



#ML libraries
# For running math with reduce operations
from tensorflow.keras import backend as K
#Dataset 
from tensorflow.keras.datasets import mnist
# For defining models
from tensorflow.keras.models import Model
# for importing all sorts of layers
from tensorflow.keras.layers import *
import tensorflow as tf
# For splitting the data to test and train subsets
from sklearn.model_selection import train_test_split
# For saving the model every epoch or so
from tensorflow.keras.callbacks import ModelCheckpoint


##### Several general functions


def choose_pickles_wildcards(max_num_pickles=100000, pickles_path=''):
    '''Load the filenames of files fullfilling a pattern /path/images/*.png'''
    files=[]
    i=1
    for file in glob.glob(pickles_path):
        files.append(file)
        if i>=max_num_pickles:
            break
        i+=1
    return files





def load_pickle(file='/lustre/astro/rlk/Movie_frames/Ramses/Sink_91/XY/1000AU/Obs_Comp/Image_Center_Primary/Time_Series/Obs_threshold/Zoom_in/'):
    '''Loads a pickle file simply'''
    information=open(file,"rb")
    result=pickle.load(information,encoding='bytes')
    information.close()
    return result
        

    
    
    
    
def image_filename_to_fps(image_filename):
    '''extract the frame,projection and sink from image filename'''
    num = re.findall(r'\d+', image_filename)
    frame=num[-2]
    projection=num[-1]
    sink=num[0]   
    return frame,projection,sink






        
def rescale_data(x,mom='0',model_name='vgg16'): 
    '''clips, rescales linearly or logarithmically the incoming images'''
    
    # Decide linear or logarithmic scaling
    if mom=='0': # I_nu
        min_rescale=1e21 #g/cm^2
        max_rescale=1e30 #g/cm^2
        mode='log'
    if mom=='1': #  v_rad
        min_rescale=-500000 #cm/s
        max_rescale =500000  #cm/s
        mode='linear'
        
    # Clip data
    x_clipped=np.copy(x)
    x_clipped[np.where(x<min_rescale)]=min_rescale
    x_clipped[np.where(x>max_rescale)]=max_rescale
    if mode=='linear':
        x_rescaled=(255*(x_clipped-min_rescale*np.ones_like(x_clipped))/(max_rescale-min_rescale))
    elif mode=='log':
        x_rescaled=(255*np.log10(x_clipped/min_rescale)/np.log10(max_rescale/min_rescale))
    x_rescaled=x_rescaled.astype("float32")
    
    
    # Preprocess images according to neural network used
    if 'DenseNet' in model_name:
        model_name_class='densenet'
    elif ('Efficient' in model_name) and ('V2' in model_name):
         model_name_class='efficientnet_v2'
    elif ('Efficient' in model_name):
         model_name_class='efficientnet'
    elif (model_name=='InceptionResNetV2'):
         model_name_class='inception_resnet_v2'
    elif (model_name=='InceptionV3'):
         model_name_class='inception_v3'
    elif (model_name=='MobileNet'):
         model_name_class='mobilenet'
    elif (model_name=='MobileNetV2'):
         model_name_class='mobilenet_v2'
    elif (model_name=='MobileNetV3Large' or model_name=='MobileNetV3Small'):
         model_name_class='mobilenet_v3'
    elif ('RegNet' in model_name):
         model_name_class='regnet'
    elif ('ResNetRS' in model_name):
         model_name_class='resnet_rs'
    elif ('ResNet' in model_name) and ('V2' in model_name):
         model_name_class='resnet_v2'
    elif ('ResNet' in model_name):
         model_name_class='resnet'
    elif ('vgg16'==model_name or 'vgg19'==model_name):
        model_name_class=model_name
    string_to_execute='x_rescaled=tf.keras.applications.'+model_name_class+'.preprocess_input(x_rescaled)'
    exec(string_to_execute)
    return x_rescaled

def load_metadata(frames,sinks,pred='roche'):
# THe options for the predicted variable are 
#"roche" for roche lobe in AU#
#"t" for time after formation in ky, 
#"disk" for disk_size in AU.
#"m1' for primary mass in Msun
#'m2' for secondary mass in Msun
# 'm' for total mass (m1+m2) in Msun
    ''' Loads the metadata of a specific frame and sink'''
    output_metadata=[]
    for i,frame in enumerate(frames):
        file='/lustre/astro/vitot/disk_pickles/analysis/sink_'+sinks[i]+'/output_'+frame[1:]+'/disk_characteristics.pkl'
        # print('trying to load metadata from path',file)
        
        data=load_pickle(file=file)
        rs,disk_size,v_phi_shellaverage,v_phi_delta,v_r_shellaverage,v_r_delta,v_z_shellaverage,v_z_delta,keplerian_v,l_outer,l_inner,gass_spin_refined,m_outer,m_inner, t_after_formation,separation,roche_r,nout,sink_tag,secondary_sink_tag,units,rsink_file,_ = data

        if pred=='roche':
            output_metadata.append(np.array(roche_r))
        elif pred=='t':
            output_metadata.append(np.array(t_after_formation))
        elif pred=='disk':
            output_metadata.append(np.array(disk_size))
        elif pred=='m1':
            output_metadata.append(np.array(3000.0*rsink_file['m'][sink_tag]))
        elif pred=='m2':
            output_metadata.append(np.array(3000.0*rsink_file['m'][secondary_sink_tag]))
        elif pred=='m':
            result=np.array(3000.0*rsink_file['m'][sink_tag])+np.array(3000.0*rsink_file['m'][secondary_sink_tag])
            output_metadata.append(result)
        # print('theoretically we just loaded the metadata')
    return np.array(output_metadata)



def does_metadata_exist(frame,sink):
    
    file='/lustre/astro/vitot/disk_pickles/analysis/sink_'+sink+'/output_'+frame[1:]+'/disk_characteristics.pkl'
    return os.path.exists(file)


def load_data(max_num_pickles, pickles_path, desired_projections,desired_sinks):
    #Filenames of all pictures
    filenames=choose_pickles_wildcards(max_num_pickles=max_num_pickles,pickles_path=pickles_path)
    
    # print('found ',len(filenames),' filenames')
    # print(filenames)
    #Initialize inputs to select samples on which training will happen
    
    frames=[] 
    projections=[]
    sinks=[]
    fnames=[]
    x=[]
    
    print('loading %d pictures:'%max_num_pickles)
    for filename in filenames:
        # print('filename:',filename)
        if not 'Sink_91/' in filename:  # Exclude low res sink 91 run
            fnames.append(filename)

            #Include only desired combinations of desired projections and desired sinks
    for filename in fnames:
        if any(desired_projection in filename for desired_projection in desired_projections):
            if any(desired_sink in filename for desired_sink in desired_sinks):
                # print(' successful filename:',filename)
                f,p,s=image_filename_to_fps(filename)
                if does_metadata_exist(f,s):
                    pic=load_pickle(filename)
                    x.append(pic)
                    frames.append(f)
                    projections.append(p)
                    sinks.append(s)
                
    if max_num_pickles>=len(np.array(sinks)):
        max_num_pickles=len(np.array(sinks))
    # print('max num pickles:',max_num_pickles)
    # print(np.shape(all_pics))
    x=np.array(x)
    x=x.reshape((max_num_pickles,800,800))
    x2=np.zeros((max_num_pickles,800,800,3))
    x2[:,:,:,0]=x2[:,:,:,1]=x2[:,:,:,2]=x
    x2=x2.astype("float32")
    return x2,frames,projections,sinks



















def model_creation(model_name,add_noise=True,noise_lvl=5,random_flip=True,architecture='1',dropout_rate=0.5,dense1_size=512,dense2_size=64):
   #Path of the saved base convolutional neural network
    base_path='/data/scratch/rami/models/large_input/'+model_name+'/bot.h5'
    model=tf.keras.Sequential() #top of the network
    bot=tf.keras.models.load_model(base_path,compile=False)
    bot.trainable=False
    
    
    if add_noise==True:
        model.add(GaussianNoise(stddev=noise_lvl))
    if random_flip==True:
        model.add(RandomFlip(mode="horizontal_and_vertical"))
    model.add(bot)
    if architecture=='1':
        model.add(SpatialDropout2D(rate=dropout_rate))
    elif architecture=='2':
        model.add(Flatten())
    model.add(Dense(dense1_size,activation='relu',kernel_initializer = tf.keras.initializers.HeNormal(), bias_initializer='zeros',    kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),    bias_regularizer=tf.keras.regularizers.L2(1e-4),    activity_regularizer=tf.keras.regularizers.L2(1e-5)))      

    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(dense2_size,activation='relu',kernel_initializer = tf.keras.initializers.HeNormal(), bias_initializer='zeros',    kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),    bias_regularizer=tf.keras.regularizers.L2(1e-4),    activity_regularizer=tf.keras.regularizers.L2(1e-5)))
    model.add(Flatten())
    model.add(Dense(1,activation=None))
    model.build((None,800,800,3))
    model.summary()
    
    return model




###############################################################################

                                #Regression 


###############################################################################




def train_regression(model_name='vgg16', add_noise=False, noise_lvl=5, random_flip=False, pickles_path='/groups/astro/rlk/rlk/Students/Sink_91_HR/movie_frame_00????/projection_?_rv.pkl',max_num_pickles=200, epochs=1000, batch_size=12, learning_rate=0.0001 ,beta_1=0.001, beta_2=0.999, epsilon=1e-7, amsgrad=False,
test_size=0.1,patience=10,dropout_rate=0.5,architecture='2',dense1_size=128,dense2_size=64,prediction='t',test_projections=['0','4'],test_sinks=['240','164'],train_projections=['0','4'],train_sinks=['240','164']): 
    
    
    
    with tf.device('GPU:3'):
         
        opt=tf.keras.optimizers.Adam(    learning_rate=learning_rate,    beta_1=beta_1,    beta_2=beta_2,    epsilon=epsilon,   amsgrad=amsgrad, name='Adam')  #-----optimizer for updating gradients-------------------
        reduction='sum_over_batch_size' # "sum_over_batch_size","sum","none"  #-----------------Reduction upon calculating loss-----------
        loss=tf.keras.losses.MeanSquaredError(reduction=reduction, name="mean_squared_error")
        
        
        
        
        
        
        
        # Training data
        x,f,p,s=load_data(max_num_pickles,pickles_path,desired_projections=train_projections,desired_sinks=train_sinks) #x=data, f=frames, p=projections, s = sinks
        x_rescaled=rescale_data(x,mom='1',model_name=model_name) # rescale data for Moment 1 or moment 0 ? 
        
        # print('now we will try to load metadata with this information:',f,s,prediction)
        y_train=load_metadata(f,s,pred=prediction)
        x_train=x_rescaled
        # Splitting data to test and training subsets
        
        x2,f_test,p_test,s_test=load_data(max_num_pickles,pickles_path,desired_projections=test_projections,desired_sinks=test_sinks) #x=data, f=frames, p=projections, s = sinks
        # print(s_test,p_test)
        x_test=rescale_data(x2,mom='1',model_name=model_name)
        # print('now we will try to load metadata with this information:',f_test,s_test,prediction)
        y_test=load_metadata(f_test,s_test,pred=prediction)
        # print('shapes time: x_train, x_test, y_train, y_test:',np.shape(x_train),np.shape(x_test),np.shape(y_train),np.shape(y_test))
        
        # Transform to tensor quantities that tensorflow understands (and computes fast because of uint8 ).
        x_test = tf.convert_to_tensor(x_test, dtype=tf.uint8)
        x_train = tf.convert_to_tensor(x_train, dtype=tf.uint8)

        # Create the model
        max_num_pickles=len(f) # How many data samples?
        steps_per_epoch=np.ceil(max_num_pickles/batch_size) #in this setting all the data are read in one epoch
        model=model_creation(model_name=model_name,add_noise=add_noise,noise_lvl=noise_lvl,random_flip=random_flip,architecture=architecture,dropout_rate=dropout_rate,dense1_size=dense1_size,dense2_size=dense2_size)

        # Compile the model (Set the tracked metrics, loss function, and way of updating weights (optimizer))
        model.compile(loss=loss,optimizer=opt)
        model.summary()

        # For faster I/O we use the tensorflow object "tf.data.Dataset" we do this for both
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        test_dataset=tf.data.Dataset.from_tensor_slices((x_test,y_test))

        # also, we can batch this object according to  the batch_size, so that it feeds info to the neural net as desired.
        train_dataset=train_dataset.batch(batch_size)
        test_dataset=test_dataset.batch(batch_size)

        # When training, the neural net might want to train through the entire dataset more than once. If one doesnt activate the "repeat" option
        # then the data will run out.
        train_dataset=train_dataset.repeat()
        test_dataset=test_dataset.repeat()

        # Here the training happens. Model parameters are updates, and after each epoch ( iteration through all data), the program validates the 
        # performance of the n.n. on images it has never been trained upon (validation_data). Training info are saved on object history.
        
        
        callback1=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=patience)
        callback2=tf.keras.callbacks.ModelCheckpoint(    filepath='data/scratch/rami/logs/',    save_weights_only=True,    monitor='val_loss',    mode='min',    save_best_only=True)
        callback3=tf.keras.callbacks.ReduceLROnPlateau(    monitor="val_loss",    factor=0.8,    patience=4,    verbose=0,    mode="auto",    min_delta=0.0001,    cooldown=0,    min_lr=0)
        callback4=tf.keras.callbacks.TerminateOnNaN()
        history=model.fit(train_dataset,steps_per_epoch=steps_per_epoch ,validation_data=test_dataset, epochs=epochs,batch_size=batch_size,validation_steps=steps_per_epoch,callbacks=[callback1,callback2,callback3,callback4])

        # save the model parameters
        extra_title=str(model_name)+'_'+str(random_flip)+'_'+str(learning_rate)+'_'+str(add_noise)+'_'+str(prediction)+'_'+str(dropout_rate)+'_'+str(dense1_size)+'_'+str(dense2_size)
        saved_weights='results_last/weights_regression_'+extra_title+'.h5'
        
        model.save(saved_weights)


        # save  the training history
        saved_history='results_last/history_regression_'+extra_title+'.npy'
        np.save(saved_history,history.history)

        # summarize history for loss
        plot_title='model, r.flip, lr, noises, variable, d.rate, d1,d2 \n'+extra_title
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(plot_title)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        save_fig_title='results_last/weights_regression_'+extra_title+'.png'
        plt.yscale('log')
        plt.savefig(save_fig_title,dpi=200)
        plt.close()


        #delete variables for memory reasons
        del model
        del train_dataset
        del test_dataset
        del x_train
        del x_test
        del y_train
        del y_test
        del history
        del f,p,s
        # Here, one can project the high dimensional space to 2 dimensions, to observe how well has the net seperated learnt to calculate distinguish-
        # able features.

        #     model_embeddings=triplet_model.layers[3].predict(x_test,verbose=1)
        #     print(model_embeddings.shape)
        #     reduced_embeddings=umap.UMAP(n_neighbors=15,min_dist=0.3,metric='correlation').fit_transform(model_embeddings)
        #     print(reduced_embeddings.shape)
        #     plt.scatter(reduced_embeddings[:,0],reduced_embeddings[:,1],c=y_test)
        #     plt.savefig('reduced_embeddings.png')
    return 0






# model_names=[  ]
# model_names=['EfficientNetB0','EfficientNetB1','EfficientNetB2','EfficientNetB3','EfficientNetB4','EfficientNetB5','EfficientNetB6','EfficientNetB7','EfficientNetV2B0','EfficientNetV2B1','EfficientNetV2B2','EfficientNetV2B3','EfficientNetV2L','EfficientNetV2M','EfficientNetV2S']
# model_names=['MobileNet','InceptionResNetV2','InceptionV3','vgg16','vgg19']
# model_names=[,]
# model_names=['MobileNetV2','ResNet152V2','ResNet50','ResNet50V2','ResNetRS101','ResNetRS152','ResNetRS200']
model_names=['MobileNetV3Large','DenseNet121', 'DenseNet169',             'DenseNet201','Xception']



predictions=['t','m1','m2','m','disk']
random_flips=[True,False]
learning_rates=[1e-3]
add_noises=[False]
architectures=['1']
dropout_rates=[0,0.25,0.5,0.75]
dense1_sizes=[64,128,256]
dense2_sizes=[32,64,128]
noise_lvl=3
max_num_pickles=1000000
test_size=0.1
batch_size=20
patience=15
beta_1=0.001
beta_2=0.999
epsilon=1e-07
amsgrad=False
epochs=50
train_projections=['projection_0','projection_2','projection_4','projection_6']
train_sinks=['Sink_49']
test_projections=['projection_2']
test_sinks=['Sink_91_HR']
# learning_rates=[0.01]


for model_name in model_names:
    for random_flip in random_flips:
        for learning_rate in learning_rates:
            for add_noise in add_noises:
                for architecture in architectures:
                    for dropout_rate in dropout_rates:
                        for dense1_size in dense1_sizes:
                            for dense2_size in dense2_sizes:
                                for prediction in predictions:
                                    try:
                                        train_regression(model_name=model_name,prediction=prediction, #model options
                                             add_noise=add_noise, noise_lvl= noise_lvl, random_flip=random_flip, # image augmentation options
                                             pickles_path='/groups/astro/rlk/rlk/Students/Sink_*/movie_frame_00????/projection_?_rv.pkl',max_num_pickles=max_num_pickles, #image directory options
                                             test_size=test_size,epochs=epochs, batch_size=batch_size,patience=patience, # training options
                                             learning_rate=learning_rate , beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,amsgrad=amsgrad,dropout_rate=dropout_rate,architecture=architecture,dense1_size=dense1_size,dense2_size=dense2_size,
                                                        train_projections=train_projections,train_sinks=train_sinks,test_projections=test_projections,test_sinks=test_sinks) # optimizer options 
                                    except:
                                        print('Something went wrong with the running of one of the models, its ok though. ')