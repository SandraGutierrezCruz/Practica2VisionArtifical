import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse


def read_csv_file(file, delim=";"): # lee los CSV con los resultados del OCR
    """

    """
    panels_info = dict() #diccionario vacio para almacenar información 
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim) # lector csv
        line_count = 0
        for row in csv_reader:
            #print(row)
            image_name = row[0] #nombre imagen 
            try:
                panel_text = row[5] # texto reconocido 
            except:
                panel_text = "" # Fallo OCR 
                
            #almacenar en el diccionario
            if panels_info.get(image_name) is None: #añade imagen si no está 
                panels_info[image_name] = [panel_text] #añadir imagen clave (nombre) valor (panel_text)
            else:
                print('image=', image_name)
                l = panels_info[image_name]
                l.append(panel_text) # añade contenido a una imagen que ya existente en el diccionario 
                panels_info[image_name] = l

            line_count += 1
    return panels_info


def levenshtein_distance(str1, str2): #calculo de la distancia de Levenshtein entre 2 strings
    """
    https://ast.wikipedia.org/wiki/Distancia_de_Levenshtein
    """
    d = dict()
    for i in range(len(str1) + 1): #inicializacion primera columna matriz 
        d[i] = dict()
        d[i][0] = i
        
    for i in range(len(str2) + 1): # incializacion primera fila matriz
        d[0][i] = i
        
    for i in range(1, len(str1) + 1): #construccion matriz utilizando formula clasica de programacion dinamica
        for j in range(1, len(str2) + 1):
            d[i][j] = min(d[i][j - 1] + 1, d[i - 1][j] + 1, d[i - 1][j - 1] + (not str1[i - 1] == str2[j - 1]))

    return d[len(str1)][len(str2)]


def plot_recognition_distance(p_gt, p): #histograma distancia levenshtein
    """
    """
    txt_distance_all = []
    for img_name in p_gt: #iterar sobre cada imagen en el ground truth
        p_info_gt = p_gt[img_name]

        if p.get(img_name) is None:  # No hay resultados OCR para esta imagen
            txt_distance_all.append(-1) # -1 marcado como no reconocido
            continue

        p_info = p[img_name] # resultados OCR para esta imagen
        # By now we assume only one detection for each image
        plate_gt = p_info_gt[0]
        
        if len(p_info) >= 1:  # al menos un resultado OCR calculo distancia con texto real
            plate = p_info[0]

            txt_distance = levenshtein_distance(plate_gt, plate)
            txt_distance_all.append(txt_distance)

    print(txt_distance_all)

    # Generar y mostrar el histograma
    plt.figure()
    #hist, bin_edges = np.histogram(np.array(txt_distance_all),  bins=8, density=False)
    #plt.step(bin_edges[:-1], hist, where='mid')
    plt.hist(np.array(txt_distance_all))
    plt.title("Distancia de Levenshtein: texto reconocido vs real")
    plt.ylabel("Núm. imágenes")
    plt.xlabel('Distancia de edición (en "número de operaciones")')
    plt.show()
#    print("hist=", hist)
#    print("bin_edges=", bin_edges)
#    print(hist[0:5].sum())

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calcula distancias de Levenshtein para un OCR')
    parser.add_argument(
        '--ocr_gt_file', default="./test_ocr_words_plain/gt.txt", help='Fichero con palabras reales (ground truth)') #archivo ground truth
    parser.add_argument(
        '--ocr_estimated_file', default="./results_ocr_words_plain/results_text_lines.txt", help='Fichero con palabras estimadas por el OCR') #archivo resultados OCR
    args = parser.parse_args()

    words_gt = read_csv_file(args.ocr_gt_file)
    print(words_gt)

    words_estimated = read_csv_file(args.ocr_estimated_file)
    print(words_estimated)

    plot_recognition_distance(words_gt, words_estimated)
