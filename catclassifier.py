import numpy as np
import matplotlib.pyplot as plt

#display defaults
plt.rcParams['figure.figsize'] = (10, 10)        
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output

import sys
caffe_root = '../' 
sys.path.insert(0, caffe_root + 'python')
import caffe
import os
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'

caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,model_weights,caffe.TEST)     

mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels
print 'mean-subtracted values:', zip('BGR', mu)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      
transformer.set_channel_swap('data', (2,1,0))  

# set the size of the input
net.blobs['data'].reshape(50,3,227, 227)  

image = caffe.io.load_image(caffe_root + 'examples/images/tigercat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)

net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  

print 'predicted class is:', output_prob.argmax()	

labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'

labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]

top_inds = output_prob.argsort()[::-1][:5]  # reverse sort 

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])

def vis_square(data):    
    # normalize data
    data = (data - data.min()) / (data.max() - data.min())
    
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))              
               + ((0, 0),) * (data.ndim - 3))  
    data = np.pad(data, padding, mode='constant', constant_values=1)
    
    # tile the filters 
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')

filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))

feat = net.blobs['conv1'].data[0, :36]
vis_square(feat)

feat = net.blobs['pool5'].data[0]
vis_square(feat)

feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)

feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
plt.show()
