
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[3]:


PATH = "data/dogbreeds/"
sz = 224
arch = resnext101_64
bs = 58


# In[4]:


label_csv = f'{PATH}labels.csv'
#makes list of rows from csv file, takes out header
n = len(list(open(label_csv)))-1
#returns 20% of the rows for use as validation set
val_idxs = get_cv_idxs(n)


# In[5]:


get_ipython().system('ls {PATH}')


# In[6]:


#read csv file
label_df = pd.read_csv(label_csv)


# In[7]:


label_df.head()


# In[8]:


label_df.pivot_table(index = 'breed', aggfunc=len).sort_values('id', ascending=False)


# In[9]:


#alows for image transformations with sides on and zooms (by up to 1.1 times) to reduce sound
tfms = tfms_from_model(arch, sz, aug_tfms = transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test',
                                   val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)


# In[10]:


fn = PATH+data.trn_ds.fnames[0]; fn


# In[11]:


img = PIL.Image.open(fn); img


# In[12]:


img.size


# In[20]:


#create dictionary that maps filenames to size
size_d = {k: PIL.Image.open(PATH+k).size for k in data.trn_ds.fnames}


# In[21]:


row_sz, col_sz = list(zip(*size_d.values()))


# In[22]:


row_sz=np.array(row_sz); col_sz = np.array(col_sz)


# In[23]:


row_sz[:5]


# In[24]:


plt.hist(row_sz)


# In[28]:


plt.hist(row_sz[row_sz<1000])


# In[29]:


plt.hist(col_sz);


# In[30]:


plt.hist(col_sz[col_sz<1000])


# In[31]:


len(data.trn_ds), len(data.test_ds)


# In[32]:


len(data.classes), data.classes[:5]


# In[33]:


def get_data(sz,bx):
       tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
       data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test',
                                          num_workers=4, val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)
       return data if sz>300 else data.resize(340, 'tmp')


# In[34]:


data = get_data(sz, bs)


# In[35]:


learn = ConvLearner.pretrained(arch, data, precompute = True)

