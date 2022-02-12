import cv2
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pixellib
from pixellib.tune_bg import alter_bg
from sklearn.cluster import KMeans
from PIL import Image
from collections import defaultdict
import webcolors
from autocrop import Cropper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd
from csv import reader


st.title('Model Pose Generator')
st.markdown("Welcome to this simple web application that returns properties such as dress, hair, skin and the dominant colours wehen a model's picture is uploaded")


def main():

    uploaded_file = st.file_uploader("Choose an image file", type="jpg")
    class_btn = st.button("Process")
    if uploaded_file is not None:    
        img = Image.open(uploaded_file)

        plt.imshow(img, interpolation='none')
        plt.axis("off")
        plt.savefig('output_image')

        st.image(img, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if uploaded_file is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):

                predict()
                model2 = KMeans(n_clusters=5)

                img2 = cv2.imread('new_pink_img.jpg')

                clt_2 = model2.fit(img2.reshape(-1, 3))
                arr= model2.cluster_centers_
                palettes = palette(arr)
                image_2 = cv2.cvtColor(palette(arr), cv2.COLOR_BGR2RGB)
                plt.imsave('scale.png',image_2)

                img3 = cv2.imread('scale.png')
                plt.imshow(img3, interpolation='none')
                st.image(img3, caption='Color scale', use_column_width=True,channels="BGR")

                im = Image.open('scale.png')
                by_color = defaultdict(int)
                l = []
                for pixel in im.getdata():
                    by_color[pixel] += 1
                x = []
                y = []
                by_color = list(by_color)

                for i in by_color:
                    i = list(i)
                    for j in range(0,3):
                        y.append(i[j])
                    s = y.copy()
                    x.append(s)
                    y.clear()
                color_list=[]
                for i in x:
                    name,hash=hex2name(i)
                    print(hex2name(i))
                    color_list.append(name)
                # color_list.remove('tan')
                x = np.array(color_list)
                # st.write(x)
                df = pd.read_csv("colors.csv")

                with open('colors.csv', 'r') as f:
                    csv_reader = reader(f)
                    final = []
                    index = []
                    count = 0

                    for row in csv_reader:
                        found = 0
                        for j in row:
                            if(j!='color1' and j!='color2' and j!='color3' and j!='color4' and j!='color5'):
                                for i in x:
                                    if(i==j):
                                        found+=1
                                        break
                        if(found>=3):
                            index.append(count)
                        count+=1

                final = index[0:10]

                with open('names.csv', 'r') as f:
                    csv_reader = reader(f)
                    csv_reader = list(csv_reader)

                for i in final:
                                
                    path = str(csv_reader[i])
                    print(path)

                    bg = path[2:-2]

                    fin = cv2.imread(bg)
                    plt.imshow(fin, interpolation='none')
                    st.image(fin, caption='Recommendations', use_column_width=True,channels="BGR")


def predict():

    img_path = "output_image.png"
    change_bg = alter_bg(model_type = "pb")
    change_bg.load_pascalvoc_model("xception_pascalvoc.pb")

    final=change_bg.color_bg(img_path,colors=(214,176,161), output_image_name="new_pink_img.jpg", detect ="person")


def bg_palette(clusters_centers):
    width=300
    palette = np.zeros((50, width, 3), np.uint8)
    steps = width/3
    for idx, centers in enumerate(clusters_centers): 
        palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
    return palette


def palette(clusters_centers):
    width=300
    palette = np.zeros((50, width, 3), np.uint8)
    steps = width/5
    for idx, centers in enumerate(clusters_centers): 
        palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
    return palette


def hex2name(c):
    h_color = '#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2]))
    try:
        nm = webcolors.hex_to_name(h_color, spec='css3')
    except ValueError as v_error:
        #print("{}".format(v_error))
        rms_lst = []
        for img_clr, img_hex in webcolors.CSS3_NAMES_TO_HEX.items():
            cur_clr = webcolors.hex_to_rgb(img_hex)
            rmse = np.sqrt(mean_squared_error(c, cur_clr))
            rms_lst.append(rmse)

        closest_color = rms_lst.index(min(rms_lst))

        nm = list(webcolors.CSS3_NAMES_TO_HEX.items())[closest_color][0]
    return nm,h_color

if __name__ == "__main__":
    main()