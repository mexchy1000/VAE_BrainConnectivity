# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import NullFormatter

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.optimizers import SGD, RMSprop

from sklearn.manifold import TSNE
from sklearn import metrics
from scipy.stats import ttest_ind


np.random.seed(1337)

df_pheno=pd.read_csv('./PhenotypeFile.csv')
df_pheno.FIQ[df_pheno.FIQ < 0] = np.nan 
Dx=np.array(df_pheno.DX_GROUP==1, dtype="float32")


#index for connectivity map
idx_triu = np.where(np.triu(np.ones((90,90)),k=1)==1)

X=[]
Y=[]
idx_abn=[]
for i, fname in enumerate(df_pheno.FILE_ID):
    fmri_raw=pd.read_table('./rois_aal/'+fname+'_rois_aal.1D')
    fmri_raw=np.asarray(fmri_raw)
    conn_real=np.corrcoef(np.transpose(fmri_raw)[:90,:])
    conn_real=np.abs(conn_real)
    if np.isnan(np.max(conn_real)): 
        print fname
        idx_abn.append(i)
        continue
    X.append(conn_real[idx_triu])
    Y.append(Dx[i])
    
# Train_Test_split
X=np.asarray(X,dtype="float32")
Y=np.asarray(Y,dtype="float32")
X_train, X_test, Y_train,Y_test= train_test_split(X, Y, test_size=0.1,random_state=42, stratify=Y)          

#New dataframe..
df_pheno=df_pheno.drop(idx_abn).reset_index()
df_pheno_asd = df_pheno.loc[df_pheno.DX_GROUP==1,:]

#TSNE
print "t-SNE modeling .. to 2D"
tsnemodel=TSNE(n_components=2, random_state=33)
X_tsne=tsnemodel.fit_transform(X_train)
plt.figure(figsize=(6,6))
im=plt.scatter(X_tsne[:,0], X_tsne[:,1],
            c=Y_train, 
            s=10, edgecolors='face', alpha=0.7)
plt.colorbar(im)


#Conditional VAE model..
batch_size = 1
original_dim = X_train.shape[1]
latent_dim = 10
intermediate_dim = 128
nb_epoch = 50
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))

h = Dense(intermediate_dim, activation='relu')(x)
h2 = Dense(intermediate_dim, activation='relu')(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_h2 = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
h_decoded2 = decoder_h2(h_decoded)
x_decoded_mean = decoder_mean(h_decoded2)

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
rmsprop=RMSprop(lr=1e-3, rho=0.9, epsilon=1e-8)
sgd = SGD(lr=1e-3, decay=0.1, momentum=0.9, nesterov=True)
vae.compile(optimizer="adadelta", loss=vae_loss)

print "train..."
vae.fit(X_train, X_train,
        validation_data=(X_test, X_test),
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size)

X_=vae.predict(X, batch_size=batch_size)
plt.figure(figsize=(5,10))
plt.subplot(2,1,1)
conn= np.zeros((90,90))
conn[idx_triu]= X[0,:]
plt.imshow(conn, vmin=-0.6, vmax=0.6)
plt.subplot(2,1,2)
conn[idx_triu]= X_[0,:]
plt.imshow(conn, vmin=-0.6, vmax=0.6)

###Encoder
encoder = Model(x, z_mean)
X_encoded = encoder.predict(X, batch_size=batch_size)

print("t-SNE Processing... : To visualize encoded data with dimension: %d"  %latent_dim)
tsnemodel=TSNE(n_components=2, random_state=33)
X_tsne=tsnemodel.fit_transform(X_encoded)

plt.figure(figsize=(6, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y,
                      s=30, edgecolor='', alpha=0.8)
plt.show()

#Generator
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_h_decoded2= decoder_h2(_h_decoded)
_x_decoded_mean = decoder_mean(_h_decoded2)
generator = Model(decoder_input, _x_decoded_mean)

#--------------------------------------------------------------
### Encoder and Feature
### Feature and ttest
df_ADIR_SOCIAL=df_pheno_asd.ADI_R_SOCIAL_TOTAL_A.copy()
df_ADIR_SOCIAL[df_pheno_asd.ADI_R_RSRCH_RELIABLE==0]=np.nan
df_ADIR_SOCIAL[df_ADIR_SOCIAL<0]=np.nan   
              
              
for i in range(latent_dim):
    ttest_result = ttest_ind(X_encoded[Y==1,i], X_encoded[Y==0, i])
    print "\nFeature"+ str(i+1)+ "...P-value =  " + str(ttest_result[1])
    
    
#Visualize two features..    
def scatter_hist2(x1,y1, x2,y2, bins =20):
    nullfmt = NullFormatter()         # no labels
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
        # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # the scatter plot:
    axScatter.scatter(x1, y1, c='red', alpha=0.5,s=50,linewidth=0)
    axScatter.scatter(x2, y2, c='blue', alpha=0.5, s=50, linewidth=0)
    
    # now determine nice limits by hand:
    axHistx.hist(x1, bins=bins,
                 color='red', alpha=0.3, linewidth=None)
    axHisty.hist(y1, bins=bins, orientation='horizontal',
                 color='red', alpha=0.3, linewidth=None)
    
    axHistx.hist(x2, bins=bins,
                 color='blue', alpha=0.3, linewidth=None)
    axHisty.hist(y2, bins=bins, orientation='horizontal',
                 color='blue', alpha=0.3, linewidth=None)
    
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    plt.show()
    
scatter_hist2(X_encoded[Y==1,7], X_encoded[Y==1,3], 
              X_encoded[Y==0,7], X_encoded[Y==0,3], bins=40)


plt.figure(figsize=(6,6))
g1 = np.random.normal(1, 0.04, size=sum(Y==0))
g2 = np.random.normal(2,0.04, size=sum(Y==1))
plt.scatter(g1, 
            X_encoded[Y==0,3], 
            c='blue',s=10, alpha=0.2)
plt.scatter(g2,
            X_encoded[Y==1,3],
            c='red',s=10, alpha=0.2)
plt.boxplot([X_encoded[Y==0,3],
            X_encoded[Y==1,3]])
plt.xlabel('Group')
plt.ylabel('ASD-related feature2 value')
plt.xlim(0.5,2.5)
plt.title("ASD-related feature2")
plt.xticks([1,2], ['NC', 'ASD'], rotation='horizontal')

plt.figure(figsize=(6,6))
g1 = np.random.normal(1, 0.04, size=sum(Y==0))
g2 = np.random.normal(2,0.04, size=sum(Y==1))
plt.scatter(g1, 
            X_encoded[Y==0,7], 
            c='blue',s=10, alpha=0.2)
plt.scatter(g2,
            X_encoded[Y==1,7],
            c='red',s=10, alpha=0.2)
plt.boxplot([X_encoded[Y==0,7],
            X_encoded[Y==1,7]])
plt.xlabel('Group')
plt.ylabel('ASD-related feature1 value')
plt.xlim(0.5,2.5)
plt.title("ASD-related feature1")
plt.xticks([1,2], ['NC', 'ASD'], rotation='horizontal')
 
#-------------------------------------------------------------
# Visualize feature

#Params
num_gen=3
df_aal=pd.read_csv('./AAL_90_BrainRegions.csv')  
colorvis={}
brainregions=list(set(df_aal.LOC))
for e, reg in enumerate(brainregions):
    colorvis[reg] = cm.jet(np.linspace(0,1,len(brainregions))[e])

    
#-----------    
#Feature 4
#-----------
feature4 = np.linspace(-1,1,num_gen)
features=np.mean(X_encoded, axis=0)
features=np.tile(features,(num_gen,1))
features[:,3] = feature4   # 

X_generated = generator.predict(features, batch_size=batch_size)

Conn_generated=[]
for i in range(num_gen):
    X_gen=np.squeeze(X_generated[i,:])
    conn=np.zeros((90,90))
    conn[idx_triu]=X_gen
    conn_flip=np.rot90(np.fliplr(conn))
    conn=conn+conn_flip
    Conn_generated.append(conn)

fig = plt.figure(figsize=(30,30), dpi=300)
for i in range(num_gen):
    ax=plt.subplot(1,3,i+1)
    ax.imshow(Conn_generated[i], vmin=0, vmax=0.7, cmap='jet')
    ax.set_xticks(range(90))
    ax.set_xticklabels(df_aal.ABBR[:90], rotation = 'vertical', fontsize=6)
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        tick.label.set_color(colorvis[df_aal.LOC[i]])
    ax.set_yticks(range(90))
    ax.set_yticklabels(df_aal.ABBR[:90], rotation = 'horizontal', fontsize=6)
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label.set_color(colorvis[df_aal.LOC[i]])
    ax.tick_params(length=0)

    
#-------------------------------    
#Feaature 8     
#feature8= norm.ppf(np.linspace(0.01, 0.99, num_gen))
feature8=np.linspace(-1,1,num_gen)
features=np.mean(X_encoded, axis=0)
features=np.tile(features,(num_gen,1))
features[:,7] = feature8 # 

X_generated = generator.predict(features, batch_size=batch_size)

Conn_generated=[]
for i in range(num_gen):
    X_gen=np.squeeze(X_generated[i,:])
    conn=np.zeros((90,90))
    conn[idx_triu]=X_gen
    conn_flip=np.rot90(np.fliplr(conn))
    conn=conn+conn_flip
    Conn_generated.append(conn)


fig = plt.figure(figsize=(30,30), dpi=300)
for i in range(num_gen):
    ax=plt.subplot(1,3,i+1)
    ax.imshow(Conn_generated[i], vmin=0, vmax=0.7, cmap='jet')
    ax.set_xticks(range(90))
    ax.set_xticklabels(df_aal.ABBR[:90], rotation = 'vertical', fontsize=6)
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        tick.label.set_color(colorvis[df_aal.LOC[i]])
    ax.set_yticks(range(90))
    ax.set_yticklabels(df_aal.ABBR[:90], rotation = 'horizontal', fontsize=6)
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label.set_color(colorvis[df_aal.LOC[i]])
    ax.tick_params(length=0)


#--------------------------
#Metrics
#-------------------------
fpr, tpr, _= metrics.roc_curve(Y==0, X_encoded[:,7])
fpr2,tpr2,_=metrics.roc_curve(Y==0, X_encoded[:,3])
plt.figure(figsize=(8,8))
plt.plot(fpr,tpr, color='green', 
         label='ASD-related feature 1, area = %0.2f)'%metrics.auc(fpr,tpr))
plt.plot(fpr2,tpr2, color='purple', 
         label='ASD-related feature 2, area = %0.2f)'%metrics.auc(fpr2,tpr2))
plt.plot([0,1],[0,1], color='black', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')


