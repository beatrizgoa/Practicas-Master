from mnist2numpy import *
import argparse
import os, struct
from array import array as pyarray
import cv2
from pylab import *
from sklearn import metrics
import matplotlib.pyplot as pl



def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def redNeuronal (input_data):
    #Se carga el entrenamiento y el test
    images, labels = load_mnist('training')
    tests, targets=load_mnist('testing')

    #Se definen variables
    MatrizTraining=[]
    MatrizLabels=[]
    MatrizTest=[]
    out=[]
    acierto=0
    i=0


    #Se crean las matrices de entrenamiento
    for image in images:
        image= cv2.resize(image,(10,10))
        image=cv2.normalize(np.float32(image),image, -1, 1, cv2.NORM_MINMAX)
        imageTraining=image.reshape(100,1)
        MatrizTraining.append(imageTraining)

    MatrizTraining=np.array(MatrizTraining)

    #Se crean las matrices de test
    for image in tests:
        image= cv2.resize(image,(10,10))
        image=cv2.normalize(np.float32(image),image, -1, 1, cv2.NORM_MINMAX)
        imageTesting=image.reshape(1,100)
        MatrizTest.append(imageTesting)

    MatrizTest=np.array(MatrizTest)

    #Se crea la matriz de las etiquetas
    for label in labels:
        a=np.zeros(10)-1.0
        a[label]=1.0
        MatrizLabels.append(np.float32(a))

    MatrizLabels=np.array(MatrizLabels)

    #Se crea la red neural
    layers = np.array([100, 70,30, 10]) #Neuronas de entrada, Neuronas de la capa oculta y neruas de salida
    nnet = cv2.ml.ANN_MLP_create() #Se crea la red
    nnet.setLayerSizes(layers)#Se le asignan las capas
    nnet.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS) #retropropagacion del arror para aprendizaje
    nnet.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    nnet.setTermCriteria((cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,300,0.075)) #Numero de iteraciones y tamano de las iteraciones(tase de aprendizaje)
    nnet.train(MatrizTraining, cv2.ml.ROW_SAMPLE, MatrizLabels) #Se entrena con los datos de entrenamiento y las etiquetas

    if len(input_data)==0:
        input_data=MatrizTest

        #Se predice y se calcula el acierto
        for test in input_data:
            out.append(int(nnet.predict(test)[0]))
            if out[i]== targets[i]:
                acierto=acierto+1
            i=i+1


        #Se calculan y muestran resultados
        lista_digitos = ['0', '1', '2','3','4','5','6','7','8','9']
        print "Numeros de salida:  {}". format(out)
        print "Resultado real: {}". format(targets.T.tolist()[0])
        print "Numero de aciertos: {}". format(acierto)

        mc = metrics.confusion_matrix(targets.T.tolist()[0], out)
        np.set_printoptions(precision=2)
        print('Matriz de confusion')
        print(mc)

        fig = plt.figure()
        pl.imshow(mc, interpolation='nearest')
        pl.title('Matriz de confusion')
        pl.colorbar()
        tick_marks = np.arange(len(lista_digitos))
        pl.xticks(tick_marks, lista_digitos)
        pl.yticks(tick_marks, lista_digitos)
        pl.ylabel('Resultado real')
        pl.xlabel('Resultado de la prediccion')
        pl.show()

    else:
        for test in input_data:
            out.append(int(nnet.predict(test)[0]))
        print out
        return out


def main():
    MatrizT=[]
    M=[]

    argumento=argparse.ArgumentParser()

    argumento.add_argument('-image', '--image', required = False,
     help = "Ruta donde reside la imagen que se quiere analizar")

    #Se le asigna el primer argumento a la ruta de entrada
    args = vars(argumento.parse_args())
    in_path = args['image']

    if in_path is None:
        redNeuronal(M)

    else:
        #se lee la iamgen
        image=cv2.imread(in_path)
        h,w=image.shape[:2]
        #resizo la iamgen porque es muy grande y sino luego no puedo mostrarla por pantalla
        image=cv2.resize(image, (w/2,h/2),interpolation=cv2.INTER_AREA)

        #pre-procesado
            #A escala de grises
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Filtro gausiano
        blur = cv2.GaussianBlur(gray_image,(5,5),0)
        #Bordes canny

        edges = cv2.Canny(blur,80,200)

        edges= cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))))
        #Extraer contornos
        ima, ctrs, hier = cv2.findContours(edges, cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)

        #Obtenemos los recangulos de cada A
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]


        for rect in rects:
            cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

            # Make the rectangular region around the digit
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

            #Se recorta la region
            crop_region = gray_image[pt1:pt1+leng, pt2:pt2+leng]

            crop_region= cv2.resize(crop_region,(10,10))
            crop_region=cv2.normalize(np.float32(crop_region),crop_region, -1, 1, cv2.NORM_MINMAX)

            imageTesting=crop_region.reshape(1,100)
            MatrizT.append(imageTesting)

        MatrizT=np.array(MatrizT)
        out=redNeuronal(MatrizT)
        i=0
        for rect in rects:
            #Draw the rectangles
            cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

            # Make the rectangular region around the digit
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

            #Se recorta la region
            crop_region = gray_image[pt1:pt1+leng, pt2:pt2+leng]

            #Se dibuja el digito correspondiente
            cv2.putText(image,str(int(out[i])),(rect[0],rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)
            i=i+1

    cv2.imshow('ventana',image)
    cv2.waitKey()



if __name__ == '__main__':
    main()