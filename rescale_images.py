import glob
import pickle
import numpy as np
import os
def load_pickle(file='/lustre/astro/rlk/Movie_frames/Ramses/Sink_91/XY/1000AU/Obs_Comp/Image_Center_Primary/Time_Series/Obs_threshold/Zoom_in/'):
    '''Loads a pickle file simply'''
    information=open(file,"rb")
    result=pickle.load(information)
    information.close()
    return result
def rescale_images(linear=True,sink_no='91',rv=True):
#     lustre/astro/rlk/Image_only_pickles/Sink_49/movie_frame_000205/projection_1_rv.pkl
    files=[]
    if rv==True:
        images_path='/lustre/astro/rlk/Image_only_pickles/Sink_'+sink_no+'/movie_frame_00????x/projection_?_rv.pkl'
    else:
        images_path='/lustre/astro/rlk/Image_only_pickles/Sink_'+sink_no+'/movie_frame_00????/projection_?.pkl'
    print(images_path)
    for file in glob.glob(images_path):
    #     print(name)
        files.append(file)
    

    i=0
    for file in files:
        #print(file)
        data=load_pickle(file=file)
        max_file=np.amax(data)
        min_file=np.amin(data)
        
        
        if i==0:
            max_=max_file
            min_=min_file
        else:
            if max_<max_file:
                max_=max_file
            if min_>min_file:
                min_=min_file
        i+=1
        #print('min,max=(%1.2e,%1.2e)'%(min_,max_))
    print('ABSOLUTE min,max=(%1.2e,%1.2e)'%(min_,max_))
    for file in files:
        img=load_pickle(file=file)
#         img=img.value
        if linear==True:
            new_image=np.round(255*(img-min_*np.ones_like(img))/(max_-min_),0)
        elif linear==False:
            new_image=np.round(255*np.log10(img/min_)/np.log10(max_/min_),0)
        if rv==True:
            specific_frame_folder ='/data/scratch/rami/images/rescaled_Sink_'+sink_no+'/movie_frame_'+file[-26:-20]+'_rv/'
        else:
            specific_frame_folder ='/data/scratch/rami/images/rescaled_Sink_'+sink_no+'/movie_frame_'+file[-23:-17]+'/'
        #print(specific_frame_folder)
        if not os.path.exists(specific_frame_folder):
            os.mkdir(specific_frame_folder)
        if rv==True:
            save_folder=specific_frame_folder+'projection_'+file[-8:-7]+'_rv.pkl'
        else:
            save_folder=specific_frame_folder+'projection_'+file[-5:-4]+'.pkl'
        #print(pickle_name)
        with open(save_folder, 'wb') as f:
            output=new_image
            pickle.dump(output, f)
            
    return 0
    

rescale_images(linear=True,sink_no='164',rv=True)
rescale_images(linear=False,sink_no='164',rv=False)
rescale_images(linear=True,sink_no='49',rv=True)
rescale_images(linear=False,sink_no='49',rv=False)