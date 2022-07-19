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
 
def choose_pickles_wildcards(max_num_pickles=200, pickles_path=''):
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
    num = re.findall(r'\d+', image_filename)
    frame=num[-2]
    projection=num[-1]
    sink=num[0]   
    return frame,projection,sink



def frames_to_classes(frames,num=13):
    length=len(frames)
    frame_min=np.amin(frames)-1
    frame_max=np.amax(frames)+1
    classes=np.linspace(frame_min,frame_max,num=num)
#     print(frame_min,frame_max)
    y=np.zeros(length)
    for i in range(len(frames)):
        for j in range(num-1):
            if frames[i]>=classes[j] and frames[i]<classes[j+1]:
                y[i]=int(j)           
    return np.array(y.astype("int"))




def triplet_loss(y_true,y_pred):
    #The output is 300 long, 100 first belong to a, 100 second elements to positive and 100 last to negative 
    anchor_out=y_pred[:,0:100]
    positive_out=y_pred[:,100:200]
    negative_out=y_pred[:,200:300]
    # Find the L1 distances between the two images features
    pos_dist=K.sum(K.abs(anchor_out-positive_out),axis=1)
    neg_dist=K.sum(K.abs(anchor_out-negative_out),axis=1)
    # Convert the L1 distance into a probability (kind of )
    probs=K.softmax((pos_dist,neg_dist),axis=0)
    
    return K.mean(K.abs(probs[0]+K.abs(1.0-probs[1])))
        
    
def data_generator(t,roche,disk,m1,m2,fnames,x_train,batch_size=64):
    while True:
        a=[]
        p=[]
        n=[]
        ii=0
        for _ in range(batch_size):
            
            metadata_exists=False
            # print('entering metadata existing checking phase')
            while not metadata_exists:
                data_triplet=random.sample(list(fnames),3)
                f0,p0,s0=image_filename_to_fps(data_triplet[0])
                
                f1,p1,s1=image_filename_to_fps(data_triplet[1])
                f2,p2,s2=image_filename_to_fps(data_triplet[2])
                if does_metadata_exist(f0,s0) and does_metadata_exist(f1,s1) and does_metadata_exist(f2,s2):
                    metadata_exists=True
            # print(data_triplet,'data triplet')
            # print(fnames)
            # print('exiting metadata existing checking phase')
            # print('chosen metadata are:',data_triplet[0],data_triplet[1],data_triplet[2])
            
            indices_a = [i for i, elem in enumerate(fnames) if ((data_triplet[0] in elem))]
            indices_p= [i for i, elem in enumerate(fnames) if ((data_triplet[1] in elem))]
            indices_n=[i for i, elem in enumerate(fnames) if ((data_triplet[2] in elem))]
            idxs=np.array([indices_a,indices_p,indices_n])
            # print('indices of 1st',indices_a)
            # print('indices of 2nd',indices_p)
            # print('indices of 3rd',indices_n)
            # print('length of matrix t',np.shape(t),len(t))
            
            # print(np.shape(t),'shape of time')
            
            t_a=t[indices_a]  #indece_a = 9000 , len(t) =1300
            t_p=t[indices_p]
            t_n=t[indices_n]
            roche_a=roche[indices_a]
            roche_p=roche[indices_p]
            roche_n=roche[indices_n]
            disk_a=disk[indices_a]
            disk_p=disk[indices_p]
            disk_n=disk[indices_n]
            m1_a=m1[indices_a]
            m1_p=m1[indices_p]
            m1_n=m1[indices_n]
            m2_a=m2[indices_a]
            m2_p=m2[indices_p]
            m2_n=m2[indices_n]
            
            m1_d1=(m1_a-m1_p)**2
            
            m1_d2=(m1_a-m1_n)**2
            m2_d1=(m2_a-m2_p)**2
            m2_d2=(m2_a-m2_n)**2
            disk_d1=(disk_a-disk_p)**2
            disk_d2=(disk_a-disk_n)**2
            roche_d1=(roche_a-roche_p)**2
            roche_d2=(roche_a-roche_n)**2
            i=0
            if m1_d1>m1_d2:
                i+=1
            elif m1_d1<m1_d2:
                i-=1
            if m2_d1>m2_d2:
                i+=1
            elif m2_d1<m2_d2:
                i-=1
            if disk_d1>disk_d2:
                i+=1
            elif disk_d1<disk_d2:
                i-=1
            if roche_d1>roche_d2:
                i+=1
            elif roche_d1<roche_d2:
                i-=1
            
            anchor=x_train[idxs[0][0],:,:,:]
            if i<=0:
                positive=x_train[idxs[1][0],:,:,:]
                negative=x_train[idxs[2][0],:,:,:]
            elif i>0:
                positive=x_train[idxs[2][0],:,:,:]
                negative=x_train[idxs[1][0],:,:,:]
                
            
            a.append(anchor)
            p.append(positive)
            n.append(negative)
            print('Successfully loaded %d outta %d triplets'%(ii+1,batch_size))
            # print('Score=%d'%(i))
            ii+=1
        yield ([np.array(a),np.array(p),np.array(n)],np.zeros((batch_size,1)).astype("float32"))
                       

            
            
            
def validation_data_generator(t,roche,disk,m1,m2,fnames,x_train,batch_size=64):
    while True:
        a=[]
        p=[]
        n=[]
        ii=0
        for _ in range(batch_size):
            
            metadata_exists=False
            # print('entering metadata existing checking phase')
            while not metadata_exists:
                data_triplet=random.sample(list(fnames),3)
                f0,p0,s0=image_filename_to_fps(data_triplet[0])
                
                f1,p1,s1=image_filename_to_fps(data_triplet[1])
                f2,p2,s2=image_filename_to_fps(data_triplet[2])
                if does_metadata_exist(f0,s0) and does_metadata_exist(f1,s1) and does_metadata_exist(f2,s2):
                    metadata_exists=True
            # print(data_triplet,'data triplet')
            # print(fnames)
            # print('exiting metadata existing checking phase')
            # print('chosen metadata are:',data_triplet[0],data_triplet[1],data_triplet[2])
            
            indices_a = [i for i, elem in enumerate(fnames) if ((data_triplet[0] in elem))]
            indices_p= [i for i, elem in enumerate(fnames) if ((data_triplet[1] in elem))]
            indices_n=[i for i, elem in enumerate(fnames) if ((data_triplet[2] in elem))]
            idxs=np.array([indices_a,indices_p,indices_n])
            # print('indices of 1st',indices_a)
            # print('indices of 2nd',indices_p)
            # print('indices of 3rd',indices_n)
            # print('length of matrix t',np.shape(t),len(t))
            
            # print(np.shape(t),'shape of time')
            
            t_a=t[indices_a]  #indece_a = 9000 , len(t) =1300
            t_p=t[indices_p]
            t_n=t[indices_n]
            roche_a=roche[indices_a]
            roche_p=roche[indices_p]
            roche_n=roche[indices_n]
            disk_a=disk[indices_a]
            disk_p=disk[indices_p]
            disk_n=disk[indices_n]
            m1_a=m1[indices_a]
            m1_p=m1[indices_p]
            m1_n=m1[indices_n]
            m2_a=m2[indices_a]
            m2_p=m2[indices_p]
            m2_n=m2[indices_n]
            
            m1_d1=(m1_a-m1_p)**2
            
            m1_d2=(m1_a-m1_n)**2
            m2_d1=(m2_a-m2_p)**2
            m2_d2=(m2_a-m2_n)**2
            disk_d1=(disk_a-disk_p)**2
            disk_d2=(disk_a-disk_n)**2
            roche_d1=(roche_a-roche_p)**2
            roche_d2=(roche_a-roche_n)**2
            i=0
            if m1_d1>m1_d2:
                i+=1
            elif m1_d1<m1_d2:
                i-=1
            if m2_d1>m2_d2:
                i+=1
            elif m2_d1<m2_d2:
                i-=1
            if disk_d1>disk_d2:
                i+=1
            elif disk_d1<disk_d2:
                i-=1
            if roche_d1>roche_d2:
                i+=1
            elif roche_d1<roche_d2:
                i-=1
            
            anchor=x_train[idxs[0][0],:,:,:]
            if i<=0:
                positive=x_train[idxs[1][0],:,:,:]
                negative=x_train[idxs[2][0],:,:,:]
            elif i>0:
                positive=x_train[idxs[2][0],:,:,:]
                negative=x_train[idxs[1][0],:,:,:]
                
            
            a.append(anchor)
            p.append(positive)
            n.append(negative)
            print('Successfully loaded %d outta %d triplets'%(ii+1,batch_size))
            # print('Score=%d'%(i))
            ii+=1
        yield ([np.array(a),np.array(p),np.array(n)],np.zeros((batch_size,1)).astype("float32"))
            
            
            
            
            
            
            
        
def rescale_data(x,mom='0',model_name='vgg16'):  #clips, rescales linearly or logarithmically the incoming images
    if mom=='0': # I_nu
        min_rescale=1e21 #g/cm^2
        max_rescale=1e30 #g/cm^2
        mode='log'
    if mom=='1': # 
        min_rescale=-500000 #cm/s
        max_rescale =500000  #cm/s
        mode='linear'
    x_clipped=np.copy(x)
    x_clipped[np.where(x<min_rescale)]=min_rescale
    x_clipped[np.where(x>max_rescale)]=max_rescale
    if mode=='linear':
        x_rescaled=(255*(x_clipped-min_rescale*np.ones_like(x_clipped))/(max_rescale-min_rescale))
    elif mode=='log':
        x_rescaled=(255*np.log10(x_clipped/min_rescale)/np.log10(max_rescale/min_rescale))
    x_rescaled=x_rescaled.astype("float32")

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

    output_metadata=[]
    
    for i,frame in enumerate(frames):
            file='/lustre/astro/vitot/disk_pickles/analysis/sink_'+sinks[i]+'/output_'+frame[1:]+'/disk_characteristics.pkl'
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
    
    return np.array(output_metadata)



def does_metadata_exist(frame,sink):
    
    file='/lustre/astro/vitot/disk_pickles/analysis/sink_'+sink+'/output_'+frame[1:]+'/disk_characteristics.pkl'
    return os.path.exists(file)





def load_relevant_data(max_num_pickles, pickles_path, desired_projections,desired_sinks):
    filenames=choose_pickles_wildcards(max_num_pickles=99999999,pickles_path=pickles_path)

    fnames_out=[]



    
    
    for filename in filenames:
        if not 'Sink_91/' in filename: #exclude lowres sink 91
            if any(desired_projection in filename for desired_projection in desired_projections): #exclude unwanted projections
                if any(desired_sink in filename for desired_sink in desired_sinks): #exclude unwanted sinks
                    f,p,s=image_filename_to_fps(filename) #calculate fps from image filename
                    if does_metadata_exist(f,s):   #exclude files which dont have metadta
                        fnames_out.append(filename)       
    if len(fnames_out)<=max_num_pickles:
        return fnames_out
    else:
        return fnames_out[:max_num_pickles]
    
def load_data(max_num_pickles, filenames):
    frames=[] 
    projections=[]
    sinks=[]
    fnames=[]
    fnames_out=[]
    x=[]
    
    
    
    for filename in filenames:
        f,p,s=image_filename_to_fps(filename)
        pic=load_pickle(filename)
        x.append(pic)
        frames.append(f)
        projections.append(p)
        sinks.append(s)
        fnames_out.append(filename)

    
    
    x=np.array(x)
    x=x.reshape((max_num_pickles,800,800))
    x2=np.zeros((max_num_pickles,800,800,3))
    x2[:,:,:,0]=x2[:,:,:,1]=x2[:,:,:,2]=x
    x2=x2.astype("float32")
    return x2,frames,projections,sinks,fnames_out


















###############################################################################

#Regression based on VGG19


###############################################################################


def model_creation(model_name,add_noise=True,noise_lvl=5,random_flip=True,architecture='1',dropout_rate=0.5,dense1_size=512,dense2_size=64,dense3_size=100):
   
    base_path='/data/scratch/rami/models/large_input/'+model_name+'/bot.h5'
    model=tf.keras.Sequential()
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
    model.add(Dense(dense3_size,activation='softmax'))
    model.build((None,800,800,3))
    model.summary()
    
    return model


def train_retrieval(model_name='vgg16', add_noise=False, noise_lvl=5, random_flip=False, pickles_path='/groups/astro/rlk/rlk/Students/Sink_91_HR/movie_frame_00????/projection_?_rv.pkl',max_num_pickles=200, epochs=1000, batch_size=12, learning_rate=0.0001 ,beta_1=0.001, beta_2=0.999, epsilon=1e-7, amsgrad=False,
test_size=0.1,patience=10,dropout_rate=0.5,architecture='2',dense1_size=128,dense2_size=64,dense3_size=100,prediction='t',                                               train_projections=[''],train_sinks=[''],test_projections=[''],test_sinks=['']): 
    with tf.device("/GPU:0"):
        # ############ STEP 1 : read data, metadata and models
        
        
        fnames=load_relevant_data(max_num_pickles,pickles_path,desired_projections=train_projections,desired_sinks=train_sinks)
        
         #TRAIN DATA
        #x, frames , projection, sink ,filenames
        x,f,p,s,fnames=load_data(max_num_pickles,fnames)
        #rescaled data
        x_rescaled=rescale_data(x,mom='1',model_name=model_name) #-----------------------------------------
        
        roche=load_metadata(f,s,pred='roche')
        t=load_metadata(f,s,pred='t')
        disk=load_metadata(f,s,pred='disk')
        m1=load_metadata(f,s,pred='m1')
        m2=load_metadata(f,s,pred='m2')
        print('uff')
        
        
        #VALIDATION DATA
        fnames2=load_relevant_data(max_num_pickles,pickles_path,desired_projections=test_projections,desired_sinks=test_sinks)
        x2,f2,p2,s2,fnames2=load_data(max_num_pickles,fnames2)
        #rescaled data
        x_rescaled2=rescale_data(x2,mom='1',model_name=model_name) #-----------------------------------------
        
        print('Shape of X_train, X_test:', np.shape(x),np.shape(x2))
        
        
        
        roche2=load_metadata(f2,s2,pred='roche')
        t2=load_metadata(f2,s2,pred='t')
        disk2=load_metadata(f2,s2,pred='disk')
        m12=load_metadata(f2,s2,pred='m1')
        m22=load_metadata(f2,s2,pred='m2')
        
        
        # Splitting data to test and training subsets
        x_train =x_rescaled
        x_test=x_rescaled2
        

        # Transform to tensor quantities that tensorflow understands (and computes fast because of uint8 ).
        x_train = tf.convert_to_tensor(x_train, dtype=tf.uint8)
        x_test = tf.convert_to_tensor(x_test, dtype=tf.uint8)

        
        
        #how many data did we find
        max_num_pickles=len(f)
        max_num_pickles2=len(f2)
        # how many steps should the network take to train on them
        steps_per_epoch=np.ceil(max_num_pickles/batch_size) #in this setting all the data are read in one epoch
        validation_steps_per_epoch=np.ceil(max_num_pickles2/batch_size) #in this setting all the data are read in one epoch
    
    
    
        #optimizer
        opt=tf.keras.optimizers.Adam(    learning_rate=learning_rate,    beta_1=beta_1,    beta_2=beta_2,    epsilon=epsilon,   amsgrad=amsgrad, name='Adam')  #------------------------
        #reduction, sum over batch_size means that the loss is per image inference.
        reduction='sum_over_batch_size' # "sum_over_batch_size","sum","none"  #----------------------------
        loss=tf.keras.losses.MeanSquaredError(reduction=reduction, name="mean_squared_error")
        
        
        
   
        

 
    #Create the model
        model=model_creation(model_name=model_name,add_noise=add_noise,noise_lvl=noise_lvl,random_flip=random_flip,architecture=architecture,dropout_rate=dropout_rate,dense1_size=dense1_size,dense2_size=dense2_size,dense3_size=dense3_size)
        triplet_model_a=Input((800,800,3))
        triplet_model_p=Input((800,800,3))
        triplet_model_n=Input((800,800,3))
        triplet_model_out=Concatenate()([model(triplet_model_a),model(triplet_model_p),model(triplet_model_n)])
        triplet_model=Model([triplet_model_a,triplet_model_p,triplet_model_n],triplet_model_out)
        triplet_model.summary()
        triplet_model.compile(loss=triplet_loss,optimizer="adam")

        
        
        
        
#         # Here the training happens. Model parameters are updates, and after each epoch ( iteration through all data), the program validates the 
#         # performance of the n.n. on images it has never been trained upon (validation_data). Training info are saved on object history.

        # callback1=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=patience)
        callback2=tf.keras.callbacks.ModelCheckpoint(    filepath='data/scratch/rami/logs/',    save_weights_only=True,    monitor='val_loss',    mode='min',    save_best_only=True)
        # callback3=tf.keras.callbacks.ReduceLROnPlateau(    monitor="val_loss",    factor=0.9,    patience=4,    verbose=0,    mode="auto",    min_delta=0.0001,    cooldown=0,    min_lr=0)
        callback4=tf.keras.callbacks.TerminateOnNaN()
        
        
        
        
        
        
        # history=triplet_model.fit(data_generator(batch_size=batch_size),steps_per_epoch=steps_per_epoch ,validation_data=validation_data_generator(batch_size=batch_size), epochs=epochs,batch_size=batch_size,validation_steps=steps_per_epoch,callbacks=[callback1,callback2,callback3,callback4])
        generator=data_generator(t=t,roche=roche,disk=disk,m1=m1,m2=m2,fnames=fnames,x_train=x_train,batch_size=batch_size)
        validation_generator=validation_data_generator(t=t2,roche=roche2,disk=disk2,m1=m12,m2=m22,fnames=fnames2,x_train=x_test,batch_size=batch_size)
        
        
        
        
        history=triplet_model.fit_generator(generator,steps_per_epoch=steps_per_epoch,validation_data=validation_generator,validation_steps=validation_steps_per_epoch,epochs=epochs,callbacks=[callback2,callback4])

        
        
        
        
        
        
        
        
        
        
        
        #SAVING
        
        
        
        
        extra_title=str(model_name)+'_'+str(random_flip)+'_'+str(learning_rate)+'_'+str(add_noise)+'_'+str(architecture)+'_'+str(dropout_rate)+'_'+str(dense1_size)+'_'+str(dense2_size)

    #     # save the model parameters
        saved_weights='results/weights_retrieval_'+extra_title+'.h5'
        model.save(saved_weights)


    #     # save  the training history
        saved_history='results/history_retrieval_'+extra_title+'.npy'
        np.save(saved_history,history.history)

    #     # summarize history for loss
        plot_title='Retrieval: model name:'+model_name+': random flip:'+str(random_flip)+': learning rate:'+str(add_noises)
        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title(plot_title)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        save_fig_title='results/weights_retrieval_'+extra_title+'.png'
        plt.yscale('log')
        plt.savefig(save_fig_title,dpi=200)
        plt.close()


    #     #delete variables for memory reasons
        del model
        
        del x_train
        del x_test
        del y_train
        del y_test
        del history
        del x
        del y
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






model_names=[  'DenseNet121', 'DenseNet169',             'DenseNet201']
# model_names=['EfficientNetB0','EfficientNetB1','EfficientNetB2','EfficientNetB3','EfficientNetB4','EfficientNetB5','EfficientNetB6','EfficientNetB7','EfficientNetV2B0','EfficientNetV2B1','EfficientNetV2B2','EfficientNetV2B3','EfficientNetV2L','EfficientNetV2M','EfficientNetV2S']
# model_names=['InceptionResNetV2','InceptionV3','MobileNet','MobileNetV2','MobileNetV3Large','ResNet152V2','ResNet50','ResNet50V2','ResNetRS101','ResNetRS152','ResNetRS200',   'vgg16','vgg19','Xception']

prediction='t'
random_flips=[True]

learning_rates=[1e-5]

add_noises=[False]

architectures=['1']

dropout_rates=[0.5]

dense1_sizes=[128,256]

dense2_sizes=[64,128]
dense3_size=100
noise_lvl=3
max_num_pickles=200
test_size=0.1
batch_size=16
patience=25
beta_1=0.001
beta_2=0.999
epsilon=1e-07
amsgrad=False
epochs=100
train_projections=['projection_0','projection_1']
train_sinks=['Sink_49']
test_projections=['projection_2']
test_sinks=['Sink_91_HR']

# learning_rates=[0.01]
train_retrieval(model_name='vgg16',prediction='t', #model options
                                     add_noise=False, noise_lvl= noise_lvl, random_flip=False, # image augmentation options
                                     pickles_path='/groups/astro/rlk/rlk/Students/Sink_*/movie_frame_00????/projection_?_rv.pkl',max_num_pickles=max_num_pickles, #image directory options
                                     test_size=0,epochs=epochs, batch_size=10,patience=patience, # training options
                                     learning_rate=1e-4 , beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,amsgrad=amsgrad,dropout_rate=5e-2,architecture='1',dense1_size=128,dense2_size=128,dense3_size=100,
                                                        train_projections=train_projections,train_sinks=train_sinks,test_projections=test_projections,test_sinks=test_sinks) # optimizer options 

# for model_name in model_names:
#     for random_flip in random_flips:
#         for learning_rate in learning_rates:
#             for add_noise in add_noises:
#                 for architecture in architectures:
#                     for dropout_rate in dropout_rates:
#                         for dense1_size in dense1_sizes:
#                             for dense2_size in dense2_sizes:
#                                 print('running!')
#                                 train_retrieval(model_name=model_name,prediction=prediction, #model options
#                                      add_noise=add_noise, noise_lvl= noise_lvl, random_flip=random_flip, # image augmentation options
#                                      pickles_path='/groups/astro/rlk/rlk/Students/Sink_91_HR/movie_frame_00????/projection_?_rv.pkl',max_num_pickles=max_num_pickles, #image directory options
#                                      test_size=test_size,epochs=epochs, batch_size=batch_size,patience=patience, # training options
#                                      learning_rate=learning_rate , beta_1=beta_1,beta_2=beta_2,epsilon=epsilon,amsgrad=amsgrad,dropout_rate=dropout_rate,architecture=architecture,dense1_size=dense1_size,dense2_size=dense2_size,dense3_size=dense3_size) # optimizer options 