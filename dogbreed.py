
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[6]:


PATH = "data/dogbreeds/"
sz = 224
arch = resnext101_64
bs = 58


# In[7]:


label_csv = f'{PATH}labels.csv'
#makes list of rows from csv file, takes out header
n = len(list(open(label_csv)))-1
#returns 20% of the rows for use as validation set
val_idxs = get_cv_idxs(n)


# In[8]:


get_ipython().system('ls {PATH}')


# In[9]:


#read csv file
label_df = pd.read_csv(label_csv)


# In[10]:


label_df.head()


# In[11]:


label_df.pivot_table(index = 'breed', aggfunc=len).sort_values('id', ascending=False)


# In[12]:


#alows for image transformations with sides on and zooms (by up to 1.1 times) to reduce sound
tfms = tfms_from_model(arch, sz, aug_tfms = transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test',
                                   val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)


# In[13]:


fn = PATH+data.trn_ds.fnames[0]; fn


# In[14]:


img = PIL.Image.open(fn); img


# In[15]:


img.size


# In[16]:


#create dictionary that maps filenames to size
size_d = {k: PIL.Image.open(PATH+k).size for k in data.trn_ds.fnames}


# In[17]:


row_sz, col_sz = list(zip(*size_d.values()))


# In[18]:


row_sz=np.array(row_sz); col_sz = np.array(col_sz)


# In[19]:


row_sz[:5]


# In[20]:


plt.hist(row_sz)


# In[21]:


plt.hist(row_sz[row_sz<1000])


# In[22]:


plt.hist(col_sz);


# In[23]:


plt.hist(col_sz[col_sz<1000])


# In[24]:


len(data.trn_ds), len(data.test_ds)


# In[25]:


len(data.classes), data.classes[:5]


# In[26]:


def get_data(sz,bx):
       tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
       data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test',
                                          num_workers=4, val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)
       return data if sz>300 else data.resize(340, 'tmp')


# In[27]:


data = get_data(sz, bs)


# In[35]:


arch=resnet50
learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[36]:


learn.fit(1e-2, 5)


# In[37]:


from sklearn import metrics


# In[38]:


data = get_data(sz, bs)


# In[43]:


arch = resnet34
learn = ConvLearner.pretrained(arch, data, precompute = True, ps=0.5)


# In[44]:


learn.precompute = False


# In[45]:


learn.fit(1e-2, 5, cycle_len=1)


# In[ ]:


learn

