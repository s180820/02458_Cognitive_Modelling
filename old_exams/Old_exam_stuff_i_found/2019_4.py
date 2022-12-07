#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from PIL import Image
import cv2
import glob 
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from numpy import linalg as LA

def read_images(folder_path):
    images = glob.glob(folder_path+"faces/*.jpg")
    images.sort() 
    img_array = []
    for fname in images:
        print(fname)
        img = cv2.imread(fname, 0)
        img_flat = img.flatten()
        img_array.append(img_flat.tolist())
    arr = np.array(img_array)
    mean_calculated = np.mean(arr)
    arr2 = arr - mean_calculated
    print(arr2.shape)
    return arr2, mean_calculated

def read_smile():
    df = pd.read_csv("smile_intensity.txt", sep=',')
    df.sort_values(by='filename')
    data = df['smile_intensity']
    return data.to_numpy()
    
def linear_model_train_and_predict(train_x, train_y, test_x, test_y):
    reg = LinearRegression().fit(train_x, train_y)
    yhat=reg.predict(test_x)
    print("Predictions: {}".format(yhat))
    print("Ground truth: {}".format(test_y))
    
    return reg

def generate_face(smile_strength, linear_model, pca_model, mean_of_picture):
    #Generating new face in PCA space
    print(linear_model.coef_)
    alpha = (smile_strength-linear_model.intercept_)/(LA.norm(linear_model.coef_)**2)
    print(alpha)
    new_face_pca = alpha*linear_model.coef_
    print(new_face_pca)
    #Transforming from PCA to pixels
    new_face = pca_model.inverse_transform(new_face_pca)+mean_of_picture
    new_face = new_face.reshape(360,260)
    #plt.imshow(new_face, cmap = "gray")
    return new_face
#=================================================

smile_idx = read_smile()
print(smile_idx)
images_array, mean = read_images("C:/Users/Cornellius/Documents/Archiwum/D/DTU/III semester/Cognitive modelling/Exam/2019/")

# %% PCA
pca = PCA(20) #55 is answer for question 2
train_x_pca = pca.fit_transform(images_array)
print(pca.explained_variance_ratio_.cumsum())


#%% Regression coefficients
model = LinearRegression().fit(train_x_pca, smile_idx)
print(model.coef_)
#linear_model = linear_model_train_and_predict(train_x_pca, train_y, test_x_pca, test_y)

#%% Generate buzia
buzia = generate_face(1.5, model, pca, mean)
plt.imshow(buzia, cmap = "gray")
# %%
