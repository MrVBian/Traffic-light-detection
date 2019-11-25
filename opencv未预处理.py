import time
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
# caffe_root = '/home/bian/caffe/' 
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

img_root_path = './images/'
img_paths = []

for i in range(1, 8):
    img_paths.append(img_root_path + str(i) + '.jpg')

for i in range(11, 17):
    img_paths.append(img_root_path + str(i) + '.jpg')

for i in range(21, 28):
    img_paths.append(img_root_path + str(i) + '.jpg')

ans = {}

start = time.time()
for img_path in img_paths:
    # image = caffe.io.load_image(img_path)

    image = cv2.imread(img_path)
    # cv2-caffe transformer
    image = image/255.
    image = image.astype('float32')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    output = net.forward()
    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
    light = output_prob.argmax()
    ans[img_path] = light
end = time.time()

for key, value in ans.items():
    print(key + ':' + str(value) )

print('len', len(img_paths))
print( end - start )
