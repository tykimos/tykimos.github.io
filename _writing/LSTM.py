import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import theano
import os

##

def create_dataset( dataset, look_back = 1 ) :
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append( dataset[ i : ( i + look_back ), 0 ])
        dataY.append( dataset[ i + look_back, 0 ] )
    return np.array( dataX ), np.array( dataY )
    

path_train = 'Data_RNN/train'

file_list_train = [ os.path.join( path_train, file ) for file in os.listdir( path_train ) if not file.startswith('.') ]

n_train = len( file_list_train )

for n in range( n_train ) :
    
    print( 'Dataset : ', str( n + 1 ), '/', str( n_train ) )
    
    F = open( file_list_train[ n ], 'r' )
    dataset_train = []

    for Fline in F.readlines() :
        try :
            A = str.split( Fline, " " )
            dataset_train = np.append( dataset_train, np.log10( float( A[ 6 ] ) ) )
        except ValueError :
            pass
    F.close()

    dataset_train = dataset_train[ : , None ]


    look_back = 24  

    trainX, trainY = create_dataset( dataset_train, look_back )

    trainX = np.reshape( trainX, ( trainX.shape[ 0 ], trainX.shape[ 1 ], 1 ) )
    
    theano.config.compute_test_value = "ignore"
    batch_size = 1
    model = Sequential()

#    model.add( LSTM( 32, input_shape = ( None, 1 ) ), stateful = True )
    model.add( LSTM( 32, batch_input_shape = ( batch_size, look_back, 1 ), stateful = True ) )
    model.add( Dropout( 0.3 ) )
    model.add( Dense( 1, activation = 'relu' ) )
    model.compile( loss = 'mean_squared_error', optimizer = 'adam' )

    model.fit( trainX, trainY, epochs = 5, batch_size = batch_size, verbose = 2 )
    model.reset_states()
    
    trainScore = model.evaluate( trainX, trainY, batch_size = batch_size, verbose = 0 )
    print( 'Train Score : ', trainScore )


    

G = open( 'Data_RNN/test/201601.txt', 'r' )
dataset_test = []
for Gline in G.readlines() :
    try :
        A = str.split( Gline, " " )
        dataset_test = np.append( dataset_test, np.log10( float( A[ 6 ] ) ) )
    except ValueError :
        pass

batch_size = 1    
dataset_test = dataset_test[ : , None ]
look_back = 24
testX, testY = create_dataset( dataset_test, look_back )
testX = np.reshape( testX, ( testX.shape[ 0 ], testX.shape[ 1 ], 1 ) )
    
testScore = model.evaluate( testX[ : 240 ], testY[ : 240 ], batch_size = batch_size, verbose = 0 )
print( 'Test Score  : ', testScore )



look_ahead = 240
testPredict = [ np.vstack( [ testX[ -1 ][ 1 : ], testY[ -1 ] ] ) ]
predictions = np.zeros( ( look_ahead , 1 ) )
for i in range( look_ahead ) :
    prediction = model.predict( np.array( [ testPredict[ -1 ] ] ), batch_size = batch_size )
    predictions[ i ] = prediction
    testPredict.append( np.vstack( [ testPredict[ -1 ][ 1 : ], prediction ] ) )
    
plt.figure(figsize=(12,5))
plt.plot( np.arange( look_ahead ), predictions, 'r' , label = "prediction" )
plt.plot( np.arange( look_ahead ), dataset_test[ 0 : look_ahead ], label = "test function" )
plt.legend()
plt.show()
