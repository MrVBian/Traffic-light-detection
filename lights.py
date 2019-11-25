import os
import cv2
import sys
# 关闭caffe log输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np

# caffe
#设置当前目录
caffe_root = '/home/bian/caffe_GPU/' 
sys.path.insert(0, caffe_root + 'python')

if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/train_squeezenet_scratch_trainval_manual_p2__iter_8000.caffemodel'):
    print ('CaffeNet found.')
else:
    print ('Downloading pre-trained CaffeNet model...')

#启用CPU
# caffe.set_mode_cpu()
# caffe.set_mode_gpu()
caffe.set_device(0)
caffe.set_mode_gpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/train_squeezenet_scratch_trainval_manual_p2__iter_8000.caffemodel'
print('------------4444------------------------------')
net = caffe.Net(model_def, 
                model_weights,
                caffe.TEST)

mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)
# print ('mean-subtracted values:', zip('BGR', mu))

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227



print('SIZE:', net.blobs['data'].shape)

img_path = "~/document/lights/green_2.jpg"

image = caffe.io.load_image(img_path)
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image

output = net.forward()
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
light = output_prob.argmax()

print('light : ', light)
