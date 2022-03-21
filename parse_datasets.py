#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave


# In[2]:


def read_csv(file):
    df = pd.read_csv(file)
    return df


# In[3]:


def save_img(np_array, name):
    imsave(name, np_array)
    


# In[4]:


def df_to_img(df, class_name):
    for i, bit in enumerate(df.columns):
        if i == 0:
            continue
        if i > 21000:
            break
            
        #print(f'packet {bit}')
        packet = df[bit].replace(1, 255).replace(0, 127).replace(-1, 0).values
        img = packet[0:1024].reshape(32, 32)
        name = f'{class_name}_{i}'
        #print(f'saving {name} ...')
        save_img(img, f'img_dataset/{class_name}/{name}.png');


# In[5]:


chat  = read_csv('datasets/chat.csv')
df_to_img(chat , 'chat')


# In[6]:


email  = read_csv('datasets/email.csv')
df_to_img(email , 'email')


# In[ ]:


#ftps  = read_csv('datasets/ftps.csv')
#df_to_img(ftps , 'ftps')


# In[ ]:


hangout  = read_csv('datasets/hangout.csv')
df_to_img(hangout , 'hangout')


# In[ ]:




