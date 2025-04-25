# @brief Tracker
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2025
#

import numpy as np
import cv2
import os


class OCRTrainingDataLoader:
    """
    Class to read cropped images from the OCR training data generated on folders (one for each char).
    """

    def __init__(self, char_size=(30,30)): #redimension pixeles 30x30 por defecto
        self.name = 'URJC-OCR-TRAIN'
        self.char_size = char_size

    def load(self, data_path): #cargar imagenes 
        """
         Given a directory where dataset is, read all the images on each folder (= char).

         :return images where images is a dictionary of lists of images where the key is the class (= char).
        """
        images = dict()

        for root, dirs, files in os.walk(data_path): #recoorrer directorio recursivamente
            for name in sorted(dirs):
                print("====> Loading ", name, " images.")
                images[name] = self.__load_images(data_path, name, self.char_size)

        return images

    def __load_images(self, data_path, char_data_dir, chars_size, show_results=False): #procesar imagenes para un solo caracter 
        """
         Given a directory where the data from a single class of char is process and crop the images.

         :return images is a list of images
        """
        images = []
        mser = cv2.MSER_create() #detector de caracteres
        
        for i, name in enumerate(sorted(os.listdir(os.path.join(data_path, char_data_dir)))):
            I = cv2.imread(os.path.join(data_path, char_data_dir, name), 0) #lectura imagen a escala de grises
            
            if not type(I) is np.ndarray:  # verificación imagen valida
                print("*** ERROR: Couldn't read image " + name)
                continue

            # Pre-process images (detect putative chars in the training images as done in detect_putative_chars)
            regions, _ = mser.detectRegions(I) #detectar regiones de caracteres
            mser_contours = [np.array(p).reshape(-1, 1, 2) for p in regions] #contornos de las regiones
            
            thresh = cv2.adaptiveThreshold(I, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) #umbralizacion adaptativa
            contours_thresh, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #detectar contornos
            
            contours = mser_contours + contours_thresh #unir contornos de las regiones y contornos umbralizados
            
            largest_contour = max(contours, key=cv2.contourArea) #contorno mas grande
            
            #x,y,w,h = r
            x, y, w, h = cv2.boundingRect(largest_contour) #rectangulo delimitador
            original_size = max(w, h)

            if (w==1) or (h==1):
                continue

            # Enlarge rectangle a percentage p on every side
            p = 0.1
            #x = max(int(round(r[0] - r[2] * p)), 0)
            y = max(int(round(r[1] - r[3] * p)), 0)
            #w = int(round(r[2] * (1. + 2.0 * p)))
            h = int(round(r[3] * (1. + 2.0 * p)))
            r = (x, y, w, h)
            new_size = max(w, h)

            if (original_size < 10) or ():
                continue

            if (x < 0) or (y < 0) or (x + w >= I.shape[1]) or (y + h >= I.shape[0]):
                continue

            Icrop = np.zeros((new_size, new_size), dtype=np.uint8)
            x_0 = int((new_size - w) / 2)
            y_0 = int((new_size - h) / 2)

            Icrop[y_0:y_0 + h, x_0:x_0 + w] = I[y:y + h, x:x + w]
            Iresize = cv2.resize(Icrop, chars_size, interpolation=cv2.INTER_NEAREST)
            images.append(Iresize)

            # Plot results
            if show_results and (i==1):
                I2 = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
                x, y, w, h = r
                cv2.rectangle(I2, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
                cv2.imshow('Letters', I2)
                cv2.imshow('crop', cv2.resize(Icrop, None, fx=4.0, fy=4.0))
                cv2.imshow('resize', cv2.resize(Iresize, None, fx=4.0, fy=4.0))
                cv2.waitKey(500)

        return images

    def show_image_examples(self, images, num_imgs_per_class=5): #visualizar ejemplos 
        Iexamples = None
        for key in images:
            examples = [img for i, img in enumerate(images[key]) if (i < num_imgs_per_class)]

            #concatena ejemplos de un mismo en una fila 
            Irow = None
            num_imgs = 0
            for e in examples:
                if Irow is None:
                    Irow = e
                else:
                    Irow = np.hstack((Irow, e))
                num_imgs += 1

                if num_imgs == num_imgs_per_class:
                    break
            #concatena ejemplos de distintos caracteres en columnas
            if Iexamples is None:
                Iexamples = Irow
            else:
                Iexamples = np.vstack((Iexamples, Irow))

        return Iexamples
