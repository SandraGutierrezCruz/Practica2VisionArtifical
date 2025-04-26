import cv2
import os
import pickle
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crea y ejecuta un detector sobre las imágenes de test')
    parser.add_argument(
        '--train_ocr_path', default="../Materiales_Práctica2/train_ocr", help='Select the training data dir for OCR')
    parser.add_argument(
        '--test_ocr_char_path', default="../Materiales_Práctica2/test_ocr_char", help='Imágenes de test para OCR de caracteres')
    parser.add_argument(
        '--test_ocr_words_path', default="../Materiales_Práctica2/test_ocr_words_plain", help='Imágenes de test para OCR con palabras completas')
    args = parser.parse_args()

    TEST_OCR_CLASSIFIER_IN_CHARS=True
    TEST_OCR_CLASSIFIER_IN_WORDS=True
    SAVED_OCR_CLF = "clasificador.pickle"
    
    # Crear el cargador de datos OCR utilizando dicha clase 
    print("Training OCR classifier ...")

    data_ocr = template_det.data_loaders.OCRTrainingDataLoader() #LINEA DE CODIGO QUE HAY QUE REVISAR 
    #if not os.path.exists(SAVED_TEXT_READER_FILE):
    if not os.path.exists(SAVED_OCR_CLF):

        # Load OCR training data (individual char images)
        print("Loading train char OCR data ...")
        
        images_dict = data_ocr.load(args.train_ocr_path)
        
        x_train = []
        y_train = []
        for letter, images in images_dict.items():
            for image in images:
                # Redimensionar de 30x30 a 1x900
                features = image.flatten().astype(np.float32) / 255.0
                x_train.append(features)
                y_train.append(letter)
        
        x_train = np.array(x)
        y_train = np.array(y)
        
        # Codificar etiquetas 
        label_encoder = sklearn.preprocessing.LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Train the OCR classifier for individual chars
        clf = sklearn.svm.SVC(kernel='linear', C=1.0, probability=True)
        clf.fit(x, y_encoded)
        
        with open(SAVED_OCR_CLF, "wb") as pickle_file:
            pickle.dump(clf, pickle_file)

    else:
        with open(SAVED_OCR_CLF, "rb") as pickle_file:
            clf = pickle.load(pickle_file)


    #prueba de caracteres individuales
    if TEST_OCR_CLASSIFIER_IN_CHARS:
        # Load OCR testing data (individual char images) in args.test_char_ocr_path
        print("Loading test char OCR data ...")
        
        images_dict = data_ocr.load(args.test_ocr_char_path)
        x_test = []
        y_test = []
        for letter, images in images_dict.items():
            for image in images:
                # Redimensionar de 30x30 a 1x900
                features = image.flatten().astype(np.float32) / 255.0
                x_test.append(features)
                y_test.append(letter)
        
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        # Codificar etiquetas
        gt_test = label_encoder.transform(y_test)
        
        print("Executing classifier in char images ...")
        estimated_test = clf.predict(x_test)
        
        # Precision clasificador OCR
        accuracy = sklearn.metrics.accuracy_score(gt_test, estimated_test)
        print("    Accuracy char OCR = ", accuracy)

    #prueba de palabras completas
    if TEST_OCR_CLASSIFIER_IN_WORDS:
        # Load full words images for testing the words reader.
        print("Loading and processing word images OCR data ...")

        # Open results file
        results_save_path = "results_ocr_words_plain"
        try:
            os.mkdir(results_save_path)
        except:
            print('Can not create dir "' + results_save_path + '"')

        results_file = open(os.path.join(results_save_path, "results_text_lines.txt"), "w")
        
        # Execute the OCR over every single image in args.test_words_ocr_path
        # POR HACER ...







