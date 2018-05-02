from docutils.nodes import image
from sensors import sensor
import numpy as np
import threading
from pyProgeo.progeo import Progeo
import cv2
import time

#TARDA 32 MINUTOS
class MyAlgorithm():

    def __init__(self, sensor):
        self.sensor = sensor
        self.imageRight=None
        self.imageLeft=None
        self.lock = threading.Lock()
        self.camLeftP=Progeo("cameras/camA.json")
        self.camRightP=Progeo("cameras/camB.json")
        self.done=False
        self.PosCamR=self.camRightP.getCameraPosition()
        self.PosCamL=self.camLeftP.getCameraPosition()
        self.value=0
        self.imageR_pad=None
        self.imageL_pad=None
        self.imageL=None
        self.imageR=None
        self.umbral=700
        self.bordesL=None
        self.bordesR=None


    def execute(self):
        #GETTING THE IMAGES

        if self.done:
            return

        if self.value==0:

            imageLeft = self.sensor.getImageLeft()
            imageRight = self.sensor.getImageRight()

            self.value=1

            # Add your code here
            #En blanco y negro
            #imageLeft = cv2.cvtColor(imageLeft, cv2.COLOR_BGR2GRAY)
            #imageRight = cv2.cvtColor(imageRight, cv2.COLOR_BGR2GRAY)

            self.imageL = imageLeft
            self.imageR = imageRight

            #self.imageLeft = imageLeft_gray
            #self.imageRight = imageRight_gray

            # Se calculan los bordes de la im con canny
            bordes_L = self.calcular_bordes(imageLeft,300,300)
            bordes_R = self.calcular_bordes(imageRight,300,300)

            self.bordesL = bordes_L   #zeros es negro y 255 blanco
            self.bordesR = bordes_R

            self.setRightImageFiltered(bordes_R)
            self.setLeftImageFiltered(bordes_L)


            # Se calcula las posiciones de los puntos de bordes
            # primer array col, segndo fila col=lista[0] row=lista[1]
            print 'voy a calcular where'
            posicionesL=np.argwhere(bordes_L>0)
            posicionesR=np.argwhere(bordes_R>0)

            print("tam_bordes_izq: {}".format(len(posicionesL)))
            print("tam_bordes_decha: {}".format(len(posicionesR)))

            # Se calculan las imagenes con zeros
            tam_bloque = 5

            print 'voy a calcular pad'
            imageR_pad = self.calcular_pad_imagenColor(imageRight,tam_bloque)
            imageL_pad = self.calcular_pad_imagenColor(imageLeft,tam_bloque)

            self.imageR_pad = imageR_pad
            self.imageL_pad = imageL_pad

            print 'voy a calcular correspondenicas'
            # # Se calcula la normade los puntos de la im izq con dcha (las imagenes son con zeros)
            # # Se calculan las correspondencias

            self.calcular_correspondencias3(posicionesR, posicionesL, tam_bloque)
            # # devulve las correspencias en la im original, sin el tam de bloque

            #print("tam correspondencias: {}".format(lista_correspondencias.shape))


            # for i in lista_correspondencias:
            #     punto=[i[1][0],i[1][1],0]
            #     self.sensor.drawPoint(punto,(0,0,0))

            #print 'voy a dibujar'

            # self.calcular_dibujo(lista_correspondencias)

            print 'he acabado de dibujar'









    def calcular_bordes(self,imagen,th1,th2):
        # Se calculan los bordes con canny
        bordes = cv2.Canny(imagen,th1,th2)
        bordes = np.array(bordes)

        return bordes



    def calcular_norma(self, bloque1,  bloque2):
        norma =cv2.norm(bloque1, bloque2)
        return norma


    def calcular_pad_imagen(self,imagen_in,tam_bloque):
        # Se le anade un margen de ceros a la imagen

        [tamx,tamy] = imagen_in.shape #xnumero de filas, y num de cols
        # MArgen izquierd
        imagen = np.insert(imagen_in, 0, np.zeros([tam_bloque,tamx]), axis=1)
        # Margen derecho
        imagen = np.insert(imagen,  tamy+tam_bloque,np.zeros([tam_bloque,tamx]), axis=1)

        # Margen superior
        imagen = np.insert(imagen, 0, np.zeros([tam_bloque,tamy+2*tam_bloque]), axis=0)
        # Margen inferior
        imagen_out = np.insert(imagen, tamx+tam_bloque,np.zeros([tam_bloque,tamy+2*tam_bloque]), axis=0)

        return imagen_out


    def calcular_pad_imagenColor(self,imagen_in,tam_bloque):
        # Se le anade un margen de ceros a la imagen
        [tamr, tamc,tamz]=imagen_in.shape

        #Arriba y abajo
        la=np.insert(imagen_in,0,np.zeros((tam_bloque,1,tamz)),axis=1)
        la=np.insert(la,(tamc+tam_bloque),np.zeros((tam_bloque,1,tamz)),axis=1)

        #Izquierda y derecha
        la=np.insert(la,0,np.zeros((tam_bloque,tamc+2*tam_bloque,tamz)),axis=0)
        imagen_out=np.insert(la,tamr+tam_bloque,np.zeros((tam_bloque,tamc+2*tam_bloque,tamz)),axis=0)

        return imagen_out


    def calcular_correspondencias3(self, kp_right, kp_left, tam_bloque):
        lista_correspondencias=[]
        cont = 0
        f = open( 'file.txt', 'a' )
        # Cojemos un punto caracteristico de la izq
        for puntoL in kp_left:

            cont = cont + 1

            if(cont % 3 )== 0:

                # Variables
                posiciones_dcha = None
                posiciones_izqda = None
                la = 0
                umbral = self.umbral
                listassd=[]


                # Bloque y punto Izquierdo
                puntoL_desp = puntoL + tam_bloque
                bloqueL = self.imageL_pad[puntoL_desp[0]-tam_bloque:puntoL_desp[0]+tam_bloque,puntoL_desp[1]-tam_bloque:puntoL_desp[1]+tam_bloque,:]

                row1,row2 = self.calcular_epipolar(puntoL)
                if row1 > row2:
                    rowBig = row1
                    rowSmall = row2
                else:
                    rowBig = row2
                    rowSmall = row1

                for puntoR in kp_right:
                    if (puntoR[0] > rowSmall - 2 and puntoR[0] < rowBig + 2):
                        puntoR_desp = puntoR + tam_bloque
                        bloqueR = self.imageR_pad[puntoR_desp[0]-tam_bloque:puntoR_desp[0]+tam_bloque,puntoR_desp[1]-tam_bloque:puntoR_desp[1]+tam_bloque,:]

                        # Se comparan los bloques de las imagenes
                        SSD = self.calcular_norma(bloqueL,  bloqueR)

                        # Se va guardando el valor mas pequeno de ssd y sus posiciones
                        if SSD < umbral:
                            la = 1
                            umbral = SSD
                            posiciones_dcha = puntoL
                            posiciones_izqda = puntoR

                # Si se ha obtenido un valor menor que el umbral, se guarda el punto
                if la == 1:
                    self.dibujar(posiciones_izqda,posiciones_dcha)
                    #lista_correspondencias.append([posiciones_izqda,posiciones_dcha])
                    #aux=self.imageR[kp[1]-tam_bloque:kp[1]+tam_bloque,kp[0]-tam_bloque:kp[0]+tam_bloque,:]
                    #self.setRightImageFiltered(aux)
                    #time.sleep(0.7) # delays for 5 seconds

            #lista_correspondencias=np.array(lista_correspondencias)



    def calcular_epipolar(self,punto):
        # pointInOpt=self.camLeftP.graficToOptical(pointIn)
        # point3d=self.camLeftP.backproject(pointInOpt)
        # projected1 = self.camRightP.project(point3d)
        # print self.camRightP.opticalToGrafic(projected1)
        # Se calcula la linea epipolar
        tmImagen = self.imageL.shape
        col=punto[1] # Este sale del np
        row=punto[0]

        pnt=[col,row,1]

        # Se calcula el primer punto del rayo
        pointInOpt=self.camLeftP.graficToOptical(pnt)
        punto3d=self.camLeftP.backproject(pointInOpt)
        projected1 = self.camRightP.project(punto3d)
        punto1_R = self.camRightP.opticalToGrafic(projected1)

        # Se calcula el segundo punto del rayo
        [c,r,w] = self.PosCamL
        pos_camaraL =[c,r,w,1]
        projected2 = self.camRightP.project(pos_camaraL)
        punto2_R = self.camRightP.opticalToGrafic(projected2)


        # Ecuacion de la linea epipolar
        [row1, colum1,h1] = punto1_R # HE CAMBIADO EL ORDEEEEEEEEN
        [row2, colum2,h2] = punto2_R

        # (column-punto1_R[1])/(punto2_R[1]-punto1_R[1]) - (row-punto1_R[0])/(punto2_R[0]-punto1_R[0]) = 0
        # row = ((column-colum1)*(row2-row1)/(colum2-colum1)) + row1

        # Para x=0
        r1 = ((0-colum1)*(row2-row1)/(colum2-colum1)) + row1
        # Para x=tamImagen[0]
        r2 = ((tmImagen[1]-colum1)*(row2-row1)/(colum2-colum1)) + row1

        return r1,r2


    def calcular_puntos(self,pto_izq,pto_dcha):
        # pointInOpt=self.camLeftP.graficToOptical(pointIn)
        # point3d=self.camLeftP.backproject(pointInOpt)
        # projected1 = self.camRightP.project(point3d)
        # print self.camRightP.opticalToGrafic(projected1)

        [r_iz,c_iz] = pto_izq
        pto_izq = [c_iz, r_iz, 1]

        [r_de,c_de] = pto_dcha
        pto_dcha = [c_de, r_de, 1]

        # Izquierda
        pointInOpt=self.camLeftP.graficToOptical(pto_izq)
        ptoizq_3d = self.camLeftP.backproject(pointInOpt)
        [c1_L,r1_L,w1_L,h1] = ptoizq_3d
        [c2_L,r2_L,w2_L] = self.PosCamL
        punto1 = [r1_L,c1_L,w1_L]
        punto2 = [r2_L,c2_L,w2_L]

        # Derecha
        pointInOpt2=self.camRightP.graficToOptical(pto_dcha)
        ptodch_3d = self.camRightP.backproject(pointInOpt2)
        [c1_R,r1_R,w1_R,h2] = ptodch_3d
        [c2_R,r2_R,w2_R] = self.PosCamR
        punto3=[r1_R,c1_R,w1_R]
        punto4 = [r2_R,c2_R,w2_R]

        return punto1, punto2, punto3, punto4


    def calcular_triangulacion2(self,punto1,punto2,punto3,punto4):

        punto1 = np.array(punto1)
        punto2 = np.array(punto2)
        punto3 = np.array(punto3)
        punto4 = np.array(punto4)

        A = punto1 - punto3
        B = punto2 - punto1
        C = punto4 - punto3

        aux1 = ((np.dot(B, B)*np.dot(C, C)) - (np.dot(C, B)*np.dot(C, B)))
        aux2 = (np.vdot(C, C))

        if aux1 == 0:
            Pa=punto1
            ma=0
        else:
            ma = ( (np.vdot(A,C)*np.vdot(C,B) - (np.vdot(A,B)*np.dot(C,C))) / aux1 )
            # Calculate the point on line 1 that is the closest point to line 2
            Pa = punto1 + (B*ma)

        if aux2 == 0:
            Pb=punto3
        else:
            mb = ( (ma*np.vdot(C, B) + np.vdot(A, C))/ aux2 )
            # Calculate the point on line 2 that is the closest point to line 1
            Pb = punto3 + (C*mb)

        P_medio=(Pa+Pb/2)
        [A,B,C] = P_medio
        exit = np.array([A,-B,-C])

        return exit




    def dibujar(self,pto_izq,pto_dcha):
        [punto1, punto2, punto3, punto4] = self.calcular_puntos(pto_izq,pto_dcha)
        punto = self.calcular_triangulacion2(punto1,punto2,punto3,punto4)
        pntt=np.array([punto[0],punto[1],punto[2]])
        #print 'punto a pintar'
        valor = self.imageL[pto_izq[0], pto_izq[1]]
        #print("valor: {}".format(valor))
        #valor=(0,0,0)
        self.sensor.drawPoint(pntt,valor)





        #pointInOpt=self.camLeftP.graficToOptical(pointIn)
        # point3d=self.camLeftP.backproject(pointInOpt)
        # projected1 = self.camRightP.project(point3d)
        # print self.camRightP.opticalToGrafic(projected1)




    def setRightImageFiltered(self, image):
        self.lock.acquire()
        size=image.shape
        if len(size) == 2:
            image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        self.imageRight=image
        self.lock.release()


    def setLeftImageFiltered(self, image):
        self.lock.acquire()
        size=image.shape
        if len(size) == 2:
            image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        self.imageLeft=image
        self.lock.release()

    def getRightImageFiltered(self):
        self.lock.acquire()
        tempImage=self.imageRight
        self.lock.release()
        return tempImage

    def getLeftImageFiltered(self):
        self.lock.acquire()
        tempImage=self.imageLeft
        self.lock.release()
        return tempImage
