#%%

import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
from imageio import imread, imsave
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import keras.backend as K
K.set_image_data_format('channels_last')
channel_axis = -1
import time

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config = config)

#%%

SLEEP_TIME = 1000
DISPLAY_ITER = 5000
MAX_ITER = 500000

MODE = 'AIA0304_to_HMI0100'
TRIAL_NAME = 'ORIGINAL_MxLr3'

INPUT1 = 'AIA0304'
INPUT2 = 'EUVI304'
OUTPUT = 'HMI0100'

ISIZE = 1024
NC_IN = 1
NC_OUT = 1
BATCH_SIZE = 1

RSUN = 392
SATURATION = 100
THRESHOLD = 10

#%%

OP1 = INPUT1 + '_to_' + OUTPUT
OP2 = INPUT2 + '_to_' + OUTPUT

IMAGE_PATH1 = './DATA/TEST/' + INPUT1 + '/*.png'
IMAGE_PATH2 = './DATA/TEST/' + INPUT2 + '/*.png'
IMAGE_PATH3 = './DATA/TEST/' + OUTPUT + '/*.png'

IMAGE_LIST1 = sorted(glob.glob(IMAGE_PATH1))
IMAGE_LIST2 = sorted(glob.glob(IMAGE_PATH2))
IMAGE_LIST3 = sorted(glob.glob(IMAGE_PATH3))

RESULT_PATH_MAIN = './RESULTS/' + TRIAL_NAME + '/'
os.mkdir(RESULT_PATH_MAIN) if not os.path.exists(RESULT_PATH_MAIN) else None

RESULT_PATH1 = RESULT_PATH_MAIN + OP1 + '/'
os.mkdir(RESULT_PATH1) if not os.path.exists(RESULT_PATH1) else None

RESULT_PATH2 = RESULT_PATH_MAIN + OP2 + '/'
os.mkdir(RESULT_PATH2) if not os.path.exists(RESULT_PATH2) else None

FIGURE_PATH_MAIN = './FIGURES/' + TRIAL_NAME + '/'
os.mkdir(FIGURE_PATH_MAIN) if not os.path.exists(FIGURE_PATH_MAIN) else None

#%%

def SCALE(DATA, RANGE_IN, RANGE_OUT):

    DOMAIN = [RANGE_IN[0], RANGE_OUT[1]]

    def INTERP(X):
        return RANGE_OUT[0] * (1.0 - X) + RANGE_OUT[1] * X

    def UNINTERP(X):
        B = 0
        if (DOMAIN[1] - DOMAIN[0]) != 0:
            B = DOMAIN[1] - DOMAIN[0]
        else:
            B =  1.0 / DOMAIN[1]
        return (X - DOMAIN[0]) / B

    return INTERP(UNINTERP(DATA))

def TUMF_VALUE(IMAGE, RSUN, SATURATION, THRESHOLD):
    VALUE_POSITIVE = 0
    VALUE_NEGATIVE = 0

    IMAGE_SCALE = SCALE(IMAGE, RANGE_IN = [0., 255.], RANGE_OUT = [-SATURATION, SATURATION])
    
    SIZE_X, SIZE_Y = IMAGE_SCALE.shape[0], IMAGE_SCALE.shape[1]
    
    for I in range(SIZE_X):
        for J in range(SIZE_Y):
            if (I-SIZE_X/2) ** 2. + (J-SIZE_Y/2) ** 2. < RSUN ** 2. :
                if IMAGE_SCALE[I, J] > THRESHOLD :
                    VALUE_POSITIVE += IMAGE_SCALE[I, J]
                elif IMAGE_SCALE[I, J] < -THRESHOLD :
                    VALUE_NEGATIVE += IMAGE_SCALE[I, J]
                else :
                    None
                    
    FACT =  (695500./RSUN) * (695500./RSUN) * 1000 * 1000 * 100 * 100
    
    FLUX_POSITIVE = VALUE_POSITIVE * FACT
    FLUX_NEGATIVE = VALUE_NEGATIVE * FACT
    FLUX_TOTAL = FLUX_POSITIVE + abs(FLUX_NEGATIVE)
    
    return FLUX_POSITIVE, FLUX_NEGATIVE, FLUX_TOTAL


#%%
        
ITER = DISPLAY_ITER
while ITER <= MAX_ITER :

    SITER = '%07d'%ITER

    MODEL_NAME = './MODELS/' + TRIAL_NAME + '/' + MODE + '/' + MODE + '_ITER' + SITER + '.h5'

    SAVE_PATH1 = RESULT_PATH1 + 'ITER' + SITER + '/'
    os.mkdir(SAVE_PATH1) if not os.path.exists(SAVE_PATH1) else None

    SAVE_PATH2 = RESULT_PATH2 + 'ITER' + SITER + '/'
    os.mkdir(SAVE_PATH2) if not os.path.exists(SAVE_PATH2) else None
    
    FIGURE_PATH = FIGURE_PATH_MAIN + 'ITER' + SITER


    EX = 0
    while EX < 1 :
        if os.path.exists(MODEL_NAME):
            print('Starting Iter ' + str(ITER) + ' ...')
            EX = 1
        else :
            print('Waiting Iter ' + str(ITER) + ' ...')
            time.sleep(SLEEP_TIME)

    MODEL = load_model(MODEL_NAME)

    REAL_A = MODEL.input
    FAKE_B = MODEL.output
    NET_G_GENERATE = K.function([REAL_A], [FAKE_B])
    def NET_G_GEN(A):
        return np.concatenate([NET_G_GENERATE([A[I:I+1]])[0] for I in range(A.shape[0])], axis=0)
    
    UTMF_REAL = []
    UTMF_FAKE = []

    for I in range(len(IMAGE_LIST1)) :
        IMG = np.float32(imread(IMAGE_LIST1[I]) / 255.0 * 2 - 1)
        REAL = np.float32(imread(IMAGE_LIST3[I]))
        DATE = IMAGE_LIST1[I][-19:-4]
        IMG.shape = (BATCH_SIZE, ISIZE, ISIZE, NC_IN)
        FAKE = NET_G_GEN(IMG)
        FAKE = ((FAKE[0] + 1) / 2.0 * 255.).clip(0, 255).astype('uint8')
        FAKE.shape = (ISIZE, ISIZE) if NC_IN == 1 else (ISIZE, ISIZE, NC_OUT)
        SAVE_NAME = SAVE_PATH1 + OP1 + '_' + DATE + '.png'
        imsave(SAVE_NAME, FAKE)

        RP, RN, RT = TUMF_VALUE(REAL, RSUN, SATURATION, THRESHOLD)        
        FP, FN, FT = TUMF_VALUE(FAKE, RSUN, SATURATION, THRESHOLD)
        
        UTMF_REAL.append(RT)
        UTMF_FAKE.append(FT)

    for J in range(len(IMAGE_LIST2)) :
        IMG = np.float32(imread(IMAGE_LIST2[J]) / 255.0 * 2 - 1)
        DATE = IMAGE_LIST2[J][-19:-4]
        IMG.shape = (BATCH_SIZE, ISIZE, ISIZE, NC_IN)
        FAKE = NET_G_GEN(IMG)
        FAKE = ((FAKE[0] + 1) / 2.0 * 255.).clip(0, 255).astype('uint8')
        FAKE.shape = (ISIZE, ISIZE) if NC_IN == 1 else (ISIZE, ISIZE, NC_OUT)
        SAVE_NAME = SAVE_PATH2 + OP2 + '_' + DATE + '.png'
        imsave(SAVE_NAME, FAKE)

    def MAKE_FIGURE():

        I1 = np.array(imread('./DATA/TEST/AIA0304/AIA0304_20170901_120005.png'))
        I2 = np.array(imread('./DATA/TEST/AIA0304/AIA0304_20170903_120005.png'))
        I3 = np.array(imread('./DATA/TEST/AIA0304/AIA0304_20170905_120005.png'))
        I4 = np.array(imread('./DATA/TEST/AIA0304/AIA0304_20170907_120005.png'))
    
        T1 = np.array(imread('./DATA/TEST/HMI0100/HMI_20170901_120130.png'))
        T2 = np.array(imread('./DATA/TEST/HMI0100/HMI_20170903_120130.png'))
        T3 = np.array(imread('./DATA/TEST/HMI0100/HMI_20170905_120130.png'))
        T4 = np.array(imread('./DATA/TEST/HMI0100/HMI_20170907_120130.png'))

        O1 = np.array(imread(SAVE_PATH1 + '/' + OP1 + '_20170901_120005.png'))
        O2 = np.array(imread(SAVE_PATH1 + '/' + OP1 + '_20170903_120005.png'))
        O3 = np.array(imread(SAVE_PATH1 + '/' + OP1 + '_20170905_120005.png'))
        O4 = np.array(imread(SAVE_PATH1 + '/' + OP1 + '_20170907_120005.png'))

        fig2 = plt.figure()

        ax211 = fig2.add_subplot(3, 4, 1)
        ax211.imshow(I1, cmap = 'gray')
        ax211.axis('off')

        ax212 = fig2.add_subplot(3, 4, 2)
        ax212.imshow(I2, cmap = 'gray')
        ax212.axis('off')

        ax213 = fig2.add_subplot(3, 4, 3)
        ax213.imshow(I3, cmap = 'gray')
        ax213.axis('off')

        ax214 = fig2.add_subplot(3, 4, 4)
        ax214.imshow(I4, cmap = 'gray')
        ax214.axis('off')

        ax221 = fig2.add_subplot(3, 4, 5)
        ax221.imshow(O1, cmap = 'gray')
        ax221.axis('off')

        ax222 = fig2.add_subplot(3, 4, 6)
        ax222.imshow(O2, cmap = 'gray')
        ax222.axis('off')

        ax223 = fig2.add_subplot(3, 4, 7)
        ax223.imshow(O3, cmap = 'gray')
        ax223.axis('off')

        ax224 = fig2.add_subplot(3, 4, 8)
        ax224.imshow(O4, cmap = 'gray')
        ax224.axis('off')

        ax231 = fig2.add_subplot(3, 4, 9)
        ax231.imshow(T1, cmap = 'gray')
        ax231.axis('off')

        ax232 = fig2.add_subplot(3, 4, 10)
        ax232.imshow(T2, cmap = 'gray')
        ax232.axis('off')

        ax233 = fig2.add_subplot(3, 4, 11)
        ax233.imshow(T3, cmap = 'gray')
        ax233.axis('off')

        ax234 = fig2.add_subplot(3, 4, 12)
        ax234.imshow(T4, cmap = 'gray')
        ax234.axis('off')

        fig2.savefig(FIGURE_PATH + '_FIGURE2.png')
        plt.close(fig2)
    
        CC = np.corrcoef(UTMF_REAL, UTMF_FAKE)[0, 1]
        fig3 = plt.figure()
        fig3.suptitle('CC : %6.3f' % (CC))
        ax3 = fig3.add_subplot(1, 1, 1)
        ax3.plot(UTMF_REAL, UTMF_FAKE, 'ro')
        fig3.savefig(FIGURE_PATH + '_FIGURE3.png')
        plt.close(fig3)


        U1 = np.array(imread('./DATA/TEST/EUVI304/EUVI304_20140604_121615.png'))
        U2 = np.array(imread('./DATA/TEST/EUVI304/EUVI304_20140607_120615.png'))
        U3 = np.array(imread('./DATA/TRAIN/AIA0304/AIA0304_20140610_120007.png'))
        U4 = np.array(imread('./DATA/TRAIN/AIA0304/AIA0304_20140613_120007.png'))
    
        D1 = np.array(imread(SAVE_PATH2 + '/' + OP2 + '_20140604_121615.png'))
        D2 = np.array(imread(SAVE_PATH2 + '/' + OP2 + '_20140607_120615.png'))
        D3 = np.array(imread('./DATA/TRAIN/HMI0100/HMI_20140610_120130.png'))
        D4 = np.array(imread('./DATA/TRAIN/HMI0100/HMI_20140613_120130.png'))

        fig4 = plt.figure()

        ax411 = fig4.add_subplot(2, 4, 1)
        ax411.imshow(U1, cmap = 'gray')
        ax411.axis('off')

        ax412 = fig4.add_subplot(2, 4, 2)
        ax412.imshow(U2, cmap = 'gray')
        ax412.axis('off')

        ax413 = fig4.add_subplot(2, 4, 3)
        ax413.imshow(U3, cmap = 'gray')
        ax413.axis('off')

        ax414 = fig4.add_subplot(2, 4, 4)
        ax414.imshow(U4, cmap = 'gray')
        ax414.axis('off')

        ax421 = fig4.add_subplot(2, 4, 5)
        ax421.imshow(D1, cmap = 'gray')
        ax421.axis('off')

        ax422 = fig4.add_subplot(2, 4, 6)
        ax422.imshow(D2, cmap = 'gray')
        ax422.axis('off')

        ax423 = fig4.add_subplot(2, 4, 7)
        ax423.imshow(D3, cmap = 'gray')
        ax423.axis('off')

        ax424 = fig4.add_subplot(2, 4, 8)
        ax424.imshow(D4, cmap = 'gray')
        ax424.axis('off')


        fig4.savefig(FIGURE_PATH + '_FIGURE4.png')
        plt.close(fig4)

    MAKE_FIGURE()

    del MODEL
    K.clear_session()

    ITER += DISPLAY_ITER
       

