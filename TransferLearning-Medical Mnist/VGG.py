import glob
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16 
from keras.layers import Flatten,Dense
from keras.models import Model
p=r"C:\Users\MohammadMahdi\Desktop\python\TransferLearning\Data\train"
paths=glob.glob(r"{}\*".format(p))
beg="C:\\Users\\MohammadMahdi\\Desktop\\python\\TransferLearning\\Data\\train\\"
#print("There are {} output classes".format(len(paths)))
classes=[]
for s in paths:
    classes.append(s[len(beg):len(s)])
#print(classes)

id={}
cnt=0
for c in classes:
    id[c]=cnt
    cnt=cnt+1
#print(id)

def preprocessing(path,output): 
    X=[]
    Y=[]
    path_of_data=glob.glob(path+'/*.jpeg')
    for i in path_of_data :
            image=load_img(i) 
            image=img_to_array(image) 
            image=image/255.0 
            X.append(image) 
            Y.append(output) 
    return np.array(X),np.array(Y)


X=np.array
Y=np.array
for c in classes:
    x,y=preprocessing(r"{}\{}".format(p,c),id[c])
    if id[c]==0:
        X=x
        Y=y
    else:
        X=np.concatenate((X,x),axis=0)
        Y=np.concatenate((Y,y),axis=0)        
#print(X.shape)
#print(Y.shape)

vgg_model = VGG16(input_shape=[64,64,3], weights='imagenet', include_top=False)

for layer in vgg_model.layers: 
   layer.trainable = False

x = Flatten()(vgg_model.output)
prediction = Dense(6, activation='softmax')(x)

model = Model(vgg_model.input, prediction)

model.summary()

model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

model.fit(X,Y,epochs=10,verbose=1)
print("Learning is done!")

pt=r"C:\Users\MohammadMahdi\Desktop\python\TransferLearning\Data\test"
Xtest=np.array
Ytest=np.array
for c in classes:
    xk,yk=preprocessing(r"{}\{}".format(pt,c),id[c])
    if id[c]==0:
        Xtest=xk
        Ytest=yk
    else:
        Xtest=np.concatenate((Xtest,xk),axis=0)
        Ytest=np.concatenate((Ytest,yk),axis=0)

def softmaxeq(lst,x):
    mx=0.0
    i=0
    c=0
    for p in lst:
        if p>mx:
            mx=p
            c=i
        i=i+1
    if c==x:
        return True
    else:
        return False
    
Ypred=model.predict(Xtest)

corr=0;
all_=Ypred.shape[0]
for i in range(all_):
    if softmaxeq(list(Ypred[i]),Ytest[i]):
        corr=corr+1
        
print("final test score is:{}%".format(100*corr/all_))