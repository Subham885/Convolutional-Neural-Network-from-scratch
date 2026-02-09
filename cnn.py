from keras.datasets import mnist
import numpy as np
from conv import conv3x3
from maxpool import MaxPool2
from softmax import Softmax

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_images = train_X[:5000]
train_labels = train_y[:5000]
test_images = test_X[:1000]
test_labels = test_y[:1000]

conv = conv3x3(8) #28x28x1 ->26x26x8
pool = MaxPool2() #26x26x8 ->13x13x8
softmax = Softmax(13*13*8,10) #13x13x8 -> 10

def forward(image,label):
    out = conv.forward((image/255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # Calculate the cross entropy loss and accuracy
    loss = -np.log(out[label]+1e-8)
    acc = 1 if np.argmax(out) == label else 0

    return out , loss , acc

def train(im,label,lr = 0.005):
    #Forward
    out,loss,acc = forward(im,label)

    
    gradient = out.copy()
    gradient[label] -= 1
    

    gradient = softmax.backprop(gradient,lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient,lr)

    return loss, acc



print('MNIST CNN initialized')

#Train CNN

for epoch in range(3):
    print('-------Epoch %d-------' %(epoch+1))

    permutation = np.random.permutation(len(train_images))
    train_images =  train_images[permutation]
    train_labels = train_labels[permutation]


    loss = 0
    num_correct = 0
    for i,(im,label) in enumerate(zip(train_images,train_labels)):
        if i>0 and i%500 == 499:
            print(f'[Step {i+1}] Past 500 Steps: Average Loss {round((loss/500),2)} | Accuracy {round(((num_correct/500)*100),2)}%')
            loss = 0
            num_correct = 0
        
        l,acc = train(im,label)
        loss += l
        num_correct += acc
    
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)
    