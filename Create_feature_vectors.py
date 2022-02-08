#operating system
import os
# AI library
import tensorflow
#normal numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
#timing output library
import time
import random
#searching nums in strings
import re
#library to manipulate simulation outputs
import yt
import pickle
import glob
# timing library
from datetime import datetime
#A.I. needed inputs
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.layers import Input
from keras.applications.vgg19 import preprocess_input as process_vgg19
from keras.applications.xception import preprocess_input as process_xception
from keras.applications.vgg16 import preprocess_input as process_vgg16
from keras.applications.densenet import preprocess_input as process_DenseNet
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
from IPython.display import Image 
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors


# Choose arbitary file names within a folder using wildcards

def choose_pickles_wildcards(max_num_pickles=200, pickles_path=''):
    files=[]
    i=1
    for file in glob.glob(pickles_path):
    #     print(name)
        files.append(file)
        if i>=max_num_pickles:
            break
        i+=1
    return files

# Choose pickle files
def choose_pickles(max_num_pickles=2000,pickles_path='/lustre/astro/rlk/Movie_frames/Ramses/Sink_91/XY/1000AU/Obs_Comp/Image_Center_Primary/Time_Series/Obs_threshold/Zoom_in/'):
    print("Function choose_pickles is running on path:",pickles_path)
    #all accepted file extensions to look for
    pickles_extensions = ['.pkl']   # case-insensitive (upper/lower doesn't matter)
    pickles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(pickles_path) for f in filenames if os.path.splitext(f)[1].lower() in pickles_extensions]
    if max_num_pickles < len(pickles):
        pickles = [pickles[i] for i in sorted(random.sample(range(len(pickles)), max_num_pickles))]
    print("keeping %d pickles." % len(pickles))
    return pickles



### Load the Neural Network from the internet or local files
def conv_model(model_name='resnet',show_model=True,use_timing=True, layer_name='flatten'):
    print("Function conv_model is running for model:",model_name," and layer:",layer_name)
    t_1=time.time()
    path='/data/scratch/rami/models/'+model_name+'/'+layer_name
    print('Loading model from path:',path)
    model = tensorflow.keras.models.load_model(path,compile=False)
    t_2=time.time()
    if model==None:
        print('Incorrect model_name fed to function conv_model')
    elif model!=None:
        print("Freezing model layers (making them untrainable)")
        for layer in model.layers:
            layer.trainable=False
    t_3=time.time()
    if show_model==True:
        model.summary()
    t_4=time.time()
    if use_timing:
        print('Loading convolutional model:',model_name,' and layer:',layer_name)
        print('%1.2f sec to load the model parameters'%(t_2-t_1))
        print('%1.2f sec to freeze the model layers'%(t_3-t_2))
        print('%1.2f sec to show the model parameters'%(t_4-t_3))
        print("\n \n")
        
    return model

 # Delete the neural network 
def delete_model(model, clear_session=True):
    '''removes model!
    '''
    print("Function delete_model is running ")
    del model
    return 0

def find_minimum_and_maximum(images_path='/lustre/astro/rlk/Image_only_pickles/Sink_164/',max_num_pickles=3072):
#     lustre/astro/rlk/Image_only_pickles/Sink_49/movie_frame_000205/projection_1_rv.pkl
    files=[]
    path=images_path+'movie_frame_00????/projection_?_rv.pkl'
    for file in glob.glob(path):
    #     print(name)
        files.append(file)
    i=0
    for file in files:
        print(file)
        with open(file, 'rb') as f:
            data = pickle.load(f,encoding='latin1')
        max_file=np.amax(result)
        min_file=np.amin(result)
        f.close()
        
        if i==0:
            max_=max_file
            min_=min_file
        else:
            if max_<max_file:
                max_=max_file
            if min_>min_file:
                min_=min_file
        i+=1
    return min_,max_

# Load an image to the neural network
def load_image(image_path,model,model_name='resnet'):
    '''Load an image an preprocess it, and return the image and the preprocessed input'''
    
#     print("Function load_image is running on path:",image_path," on model :",model_name)
    img = image.load_img(image_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if model_name=='vgg16':  
        x=process_vgg16(x)
    elif model_name=='xception':
        x=process_xception(x)
    elif model_name=='vgg19':
        x=process_vgg19(x) 
    else:
        print('Incorrect  model_name fed to function load_image')
    
    return img, np.array(x)


#Loads a 2D array from a pickle file, and preprocesses it according to the neural netowrk chosen. Returns the image and the network input
def load_img_from_pickles(pickle_file='/lustre/astro/rlk/Movie_frames/Ramses/Sink_91/XY/1000AU/Obs_Comp/Image_Center_Primary/Time_Series/Obs_threshold/Zoom_in/'\
                          ,model_name='vgg16'):
#     print("Function load_img_from_pickles is running on path:",pickle_file," on model :",model_name)
    file = open(pickle_file, "rb")
    image = pickle.load(file)
    image=np.array(image,dtype=np.uint8)
    
    #X, Y, image = pickle.load(file)
    file.close()
    shape_image=np.shape(image)
    image_rgb=np.zeros(shape=(shape_image[0],shape_image[1],3))
    image_rgb[:,:,0]=image
    image_rgb[:,:,1]=image
    image_rgb[:,:,2]=image
   
    image_rgb=tensorflow.keras.preprocessing.image.smart_resize(image_rgb,size=(224,224))
    
   
    image_before_preprocess=np.expand_dims(image_rgb, axis=0)
    if model_name=='vgg16':  
        x=process_vgg16(image_before_preprocess)
    elif model_name=='xception':
        x=process_xception(image_before_preprocess)
    elif model_name=='vgg19':
        x=process_vgg19(image_before_preprocess) 
    
    
    return image_rgb,x






# pickles contains the paths of the pickles containing the 2D image. This code extracts the feature vector and returns it.
def pickles_to_conv_features(pickles,model,model_name,batch_size=32):
# This function is used in calculate_features.
    tic=time.time()
    every=500
    features = []
    X_train=[]
    for i, pickle_file in enumerate(pickles):

        img,x = load_img_from_pickles(pickle_file=pickle_file,model_name=model_name)
        X_train.append(x)
        
        # Timing output
        if i % every == 0:
                if i==0:
                    toc=time.time()
                elap = tic-toc;
                toc = time.process_time()
                remaining_time=(len(pickles)-i)*elap/every
                hours=remaining_time//3600
                minutes=remaining_time//60
                seconds=remaining_time%60
                print("loading image %d / %d. Time/%d pics: %4.4f seconds." % (i, len(pickles),every,elap))
                print('Remaining time: %d hours %d minutes %d sec.' %(hours,minutes,seconds))

                tic = time.process_time()
    X_train=np.reshape(X_train,(len(pickles),224,224,3))
    XX = np.copy(X_train)
    with tensorflow.device('/cpu:0'):
        feat = model.predict(XX,batch_size=batch_size)
    features=feat.reshape(len(pickles),-1)   
    print('finished extracting features for %d images' % len(pickles))
    print("\n \n")
    return np.array(features)



# pickles contains the paths of the pickles containing the 2D image. This code extracts the feature vector and returns it.
def pickle_to_conv_features(pickle,model,model_name):
    img,x = load_img_from_pickles(pickle_file=pickle,model_name=model_name)

    feat = model.predict(x)
   
    feature=feat.flatten()
    
    
    return feature




os.environ['CUDA_VISIBLE_DEVICES']='-1'
def calculate_features(pickles_path='/groups/astro/rami/images/',max_num_images=20,model_name='vgg16', layer_name='fc2' \
                       ,use_PCA=True,n_components=10, \
                       display_output=False,use_timing=True, \
                       save_features=True, features_folder='/lustre/hpc/astro/rami/features/', batch_size=32):


    print("Extracting features of %d projections \n which are in folder %s, \n using model %s \n. \n PCA used?:%s \n  "%(max_num_images,pickles_path,model_name,use_PCA))


    ### Load model from folder
    if use_timing:
        t0 = time.time()
    model=conv_model(model_name=model_name,layer_name=layer_name)
    print('step 1: loaded model')
    model.summary()
    if use_timing:
        t1 = time.time()


    ### Randomly choose max_num_images amount of pickle files from the subfolders of pickles_path.
    pickles=choose_pickles_wildcards(max_num_pickles=max_num_images,pickles_path=pickles_path)
    print("pickles shape:",np.shape(pickles))
    print('step 2: chose filenames:')
    for i,j in enumerate(pickles):
        if i%1==0:
            print(i,' name of pickle:',j)
    if use_timing:
        t2 = time.time()
        
        
        
    ### Extract features from those files
    
    features=pickles_to_conv_features(pickles=pickles,model=model,model_name=model_name,batch_size=batch_size)
    
    print("Shape of features:",np.shape(features))
    print('\n')
#     for _ in range(len(pickles)):
#         featt=features[_,:]
#         print("Shape of feature",np.shape(featt))
#         print('minimum of feature:',np.amin(featt))
#         print('maximum of feature:',np.amax(featt))
#         print('average of feature:',np.average(featt))
#         print('standard deviation of feature:',np.std(featt))
#         print("\n \n")



    # Possibly reduce dimensionality of data
    if use_timing:
        t3 = time.time()
    if use_PCA==True:
        full_features=np.vstack((feature,features))
        reduced_full_features=reduce_PCA(full_features,n_components=n_components)
        reduced_feature=reduced_full_features[0,:]
        reduced_features=reduced_full_features[1:,:]
    elif use_PCA==False:
        reduced_features=features
    if use_timing:
        t4 = time.time()    


    ### Save the outputs.
    if save_features:
        print('saving under ',features_folder)
        reduced_features=np.array(reduced_features)
        for i in range(len(pickles)):
            name=pickles[i]
            if 'rv' in name:
                rv=True
            else:
                rv=False
            specific_frame_folder1 ='/data/scratch/rami/features/Observations/'+model_name+'/'+layer_name
            if not os.path.exists(specific_frame_folder1):
                os.mkdir(specific_frame_folder1)
            print(name)
            num = re.findall(r'\d+', name)
            moment=num[1]
            print('name :%s corresponds to moment:%s'%(name,moment))
            pickle_name =specific_frame_folder1+'/moment_'+moment+'.pkl'
            
#             print('rv=',rv)
#             pickle_name='
            with open(pickle_name, 'wb') as f:
                output=reduced_features[i,:]
                pickle.dump(output, f)
    if use_timing:
        t5=time.time()



    # Timing outputs
    ###########################################################################
    if use_timing:
        Dt1=t1-t0
        Dt2=t2-t1
        Dt3=t3-t2
        Dt4=t4-t3
        Dt5=t5-t4
        print('\n')
        print('\n')
        print('Dt1=%3.3f s:Model parameter loading: '%Dt1)

        print('Dt2=%3.3f s:Choosing the images from the folder'%Dt2)


        print('Dt3=%3.3f s:1) Image preprocessing and 2)loading of feature calculator'%Dt3)
        print('Dt4=%3.3f s:PCA'%Dt4)
        print('Dt5=%3.3f s:Saving '%Dt5)
        print('\n')
        delete_model(model=model)
    return 0


# ################            VGG16
calculate_features(pickles_path='/data/scratch/rami/Moments/mom?_?x_y_b_a_??_??_??_??.pkl',max_num_images=10,\
                 model_name='vgg19',layer_name='block5_pool',\
                 use_PCA=False,n_components=100,\
                 display_output=True,use_timing=True,\
                 save_features=True, features_folder='/data/scratch/rami/',batch_size=32)
calculate_features(pickles_path='/data/scratch/rami/Moments/mom?_?x_y_b_a_??_??_??_???.pkl',max_num_images=10,\
                 model_name='vgg19',layer_name='block5_pool',\
                 use_PCA=False,n_components=100,\
                 display_output=True,use_timing=True,\
                 save_features=True, features_folder='/data/scratch/rami/',batch_size=32)


calculate_features(pickles_path='/data/scratch/rami/Moments/mom?_?x_y_b_a_??_??_??_??.pkl',max_num_images=10,\
                 model_name='vgg19',layer_name='block4_pool',\
                 use_PCA=False,n_components=100,\
                 display_output=True,use_timing=True,\
                 save_features=True, features_folder='/data/scratch/rami/',batch_size=32)
calculate_features(pickles_path='/data/scratch/rami/Moments/mom?_?x_y_b_a_??_??_??_???.pkl',max_num_images=10,\
                 model_name='vgg19',layer_name='block4_pool',\
                 use_PCA=False,n_components=100,\
                 display_output=True,use_timing=True,\
                 save_features=True, features_folder='/data/scratch/rami/',batch_size=32)


calculate_features(pickles_path='/data/scratch/rami/Moments/mom?_?x_y_b_a_??_??_??_??.pkl',max_num_images=10,\
                 model_name='vgg19',layer_name='fc1',\
                 use_PCA=False,n_components=100,\
                 display_output=True,use_timing=True,\
                 save_features=True, features_folder='/data/scratch/rami/',batch_size=32)
calculate_features(pickles_path='/data/scratch/rami/Moments/mom?_?x_y_b_a_??_??_??_???.pkl',max_num_images=10,\
                 model_name='vgg19',layer_name='fc1',\
                 use_PCA=False,n_components=100,\
                 display_output=True,use_timing=True,\
                 save_features=True, features_folder='/data/scratch/rami/',batch_size=32)


calculate_features(pickles_path='/data/scratch/rami/Moments/mom?_?x_y_b_a_??_??_??_??.pkl',max_num_images=10,\
                 model_name='vgg19',layer_name='fc2',\
                 use_PCA=False,n_components=100,\
                 display_output=True,use_timing=True,\
                 save_features=True, features_folder='/data/scratch/rami/',batch_size=32)
calculate_features(pickles_path='/data/scratch/rami/Moments/mom?_?x_y_b_a_??_??_??_???.pkl',max_num_images=10,\
                 model_name='vgg19',layer_name='fc2',\
                 use_PCA=False,n_components=100,\
                 display_output=True,use_timing=True,\
                 save_features=True, features_folder='/data/scratch/rami/',batch_size=32)


# calculate_features(pickles_path='/data/scratch/rami/images/rescaled_Sink_91/movie_frame_00????/projection_?_rv.pkl',max_num_images=20000,\
#                  model_name='vgg16',layer_name='block4_pool',\
#                  use_PCA=False,n_components=100,\
#                  display_output=True,use_timing=True,\
#                  save_features=True, features_folder='/data/scratch/rami/features/Sink_91/',batch_size=32)


# calculate_features(pickles_path='/data/scratch/rami/images/rescaled_Sink_91/movie_frame_00????/projection_?_rv.pkl',max_num_images=20000,\
#                  model_name='vgg16',layer_name='fc1',\
#                  use_PCA=False,n_components=100,\
#                  display_output=True,use_timing=True,\
#                  save_features=True, features_folder='/data/scratch/rami/features/Sink_91/',batch_size=32)

# calculate_features(pickles_path='/data/scratch/rami/images/rescaled_Sink_91/movie_frame_00????/projection_?_rv.pkl',max_num_images=20000,\
#                  model_name='vgg16',layer_name='fc2',\
#                  use_PCA=False,n_components=100,\
#                  display_output=True,use_timing=True,\
#                  save_features=True, features_folder='/data/scratch/rami/features/Sink_91/',batch_size=32)




# # ################            VGG19
# calculate_features(pickles_path='/data/scratch/rami/images/rescaled_Sink_91/movie_frame_00????/projection_?_rv.pkl',max_num_images=20000,\
#                  model_name='vgg19',layer_name='block5_pool',\
#                  use_PCA=False,n_components=100,\
#                  display_output=True,use_timing=True,\
#                  save_features=True, features_folder='/data/scratch/rami/features/Sink_91/',batch_size=32)


# calculate_features(pickles_path='/data/scratch/rami/images/rescaled_Sink_91/movie_frame_00????/projection_?_rv.pkl',max_num_images=20000,\
#                  model_name='vgg19',layer_name='block4_pool',\
#                  use_PCA=False,n_components=100,\
#                  display_output=True,use_timing=True,\
#                  save_features=True, features_folder='/data/scratch/rami/features/Sink_91/',batch_size=32)


# calculate_features(pickles_path='/data/scratch/rami/images/rescaled_Sink_91/movie_frame_00????/projection_?_rv.pkl',max_num_images=20000,\
#                  model_name='vgg19',layer_name='fc1',\
#                  use_PCA=False,n_components=100,\
#                  display_output=True,use_timing=True,\
#                  save_features=True, features_folder='/data/scratch/rami/features/Sink_91/',batch_size=32)

# calculate_features(pickles_path='/data/scratch/rami/images/rescaled_Sink_91/movie_frame_00????/projection_?_rv.pkl',max_num_images=20000,\
#                  model_name='vgg19',layer_name='fc2',\
#                  use_PCA=False,n_components=100,\
#                  display_output=True,use_timing=True,\
#                  save_features=True, features_folder='/data/scratch/rami/features/Sink_91/',batch_size=32)
