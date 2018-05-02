from sensors import sensor
import numpy as np
import threading
import cv2


class MyAlgorithm():

    desviacion_anterior = 0
    error_prev = 0
    integral_prev_vel = 0
    integral_prev_giro = 0
    giro_prev = 0.001

    def __init__(self, sensor):
        self.sensor = sensor
        self.imageRight=None
        self.imageLeft=None
        self.lock = threading.Lock()
        self.errores = np.ones(20)
        self.errores = self.errores * 320



    def execute(self):
        #GETTING THE IMAGES
        imageLeft = self.sensor.getImageLeft()
        imageRight = self.sensor.getImageRight()


        try:

            result, mask, contorno =self.calcular_carretera(imageLeft)

            centroides, result=self.calcular_centroides2(result, mask)
            ncentroides = len(centroides)

            if ncentroides>1:

                # Calculamos el error para el pid de giro
                error=self.calculamos_error(centroides, result.shape)
                centroides_perdidos = 3 - ncentroides

                if error >0:
                    signo = 1
                else:
                    signo = -1

                if centroides_perdidos > 0:
                    error = (error + centroides_perdidos * signo * 250) / (centroides_perdidos+1)

                giro =self.calcular_pid_giro(0.3095,0.000014,0.00014,error) #kp,ki,kg, error
                self.giro_prev = giro

                velocidad=self.calcular_pid_velocidad2(error) #kp,ki,kd, error


                print ("\n\n\n")
                print ("Error:"+str(error)+" Giro:"+str(giro)+" Velocidad:"+str(velocidad))
                self.sensor.setV(velocidad*0.01) #Velocidad
                self.sensor.setW(giro*0.1) #giro #izquierdo positivo; derecha negativo

            else:
                self.sensor.setV(0)
                self.sensor.setW(self.giro_prev*0.1)

            self.setLeftImageFiltered(result)



        except: #si no hay linea que gire
            self.sensor.setV(0) #Velocidad
            self.sensor.setW(self.giro_prev*0.1) #giro



    def calcular_carretera(self, imageLeft):

        # Pasamos a HSV
        hsvimage = cv2.cvtColor(imageLeft,cv2.COLOR_BGR2HSV)
        # Definimos los rangos de color de rojo
        lower_red = np.array([100,50,0],np.uint8)
        upper_red = np.array([130,255,255], np.uint8)
        # Calculamos la mascara
        mask = cv2.inRange(hsvimage, lower_red, upper_red)
        # Cogemos los contornos de la mascara con las carreteras
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # Buscamos el contorno mas grande y nos lo quedamos
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]

        # Obtemos la imagen a partir de la mascara con solo la carretera
        mask2 = np.zeros(imageLeft.shape[:2], dtype="uint8")
        cv2.drawContours(mask2, [cnt], -1,  255,  -1)
        result = cv2.bitwise_and(imageLeft, imageLeft, mask=mask2)

        #devolvemos
        return result, mask2, cnt





    def calcular_centroides2(self, imagen, mask):

        ## DIBUJAMOS LINEAS EN LA IMAGEN PARA CREAR VARIOS CENTROIDES
        tam=imagen.shape #0 es la vestical, eL alto; 1 es la horizonatal, ancho
        # Alto de las distintas lineas
        #alto0 = int(round(0.62*tam[0]))
        alto1 = int(round(0.65*tam[0]))
        alto2 = int(round(0.7*tam[0]))
        alto3 = int(round(0.75*tam[0]))
        alto4 = int(round(0.8*tam[0]))
        #alto5 = int(round(0.9*tam[0]))
        # Dibujamos las lineas
        #cv2.line(mask, (0, alto0),(tam[1], alto0), (0,0,0), 4)
        cv2.line(mask, (0, alto1),(tam[1], alto1), (0,0,0), 4)
        cv2.line(mask, (0, alto2),(tam[1], alto2), (0,0,0), 4)
        cv2.line(mask, (0, alto3),(tam[1], alto3), (0,0,0), 4)
        cv2.line(mask, (0, alto4),(tam[1], alto4), (0,0,0), 4)
        #cv2.line(mask, (0, alto5),(tam[1], alto5), (0,0,0), 4)

        # Convolucionamos la imagen con la mascara de la carretera
        imagen = cv2.bitwise_and(imagen, imagen, mask=mask)

        #Se busca contornos
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # Descartamos el primer contorno - Ya que no sabemos que va a ser
        contours.pop(0)
        contours.pop(len(contours)-1)
        # En este for de dibujamos los contornos
        #for c in contours:
        #    x,y,w,h = cv2.boundingRect(c)
        #    cv2.rectangle(imagen,(x, y), (x + w, y + h), (0, 0, 255), 2)

        # Estraemos los centroides de los contornos
        moments  = [cv2.moments(c) for c in contours]
        centroids = []
        for m in moments:
            if m['m00'] != 0:
                centroids.append( (int(round(m['m10']/m['m00'])),int(round(m['m01']/m['m00']))))
            else:
                print("Un divisor fue 0")

        for c in centroids:
            # I draw a black little empty circle in the centroid position
            cv2.circle(imagen,c,5,(0,255,0))

        #El primer centroide de la lista es el mas arriba, y el ultimo el mas abajo


        return centroids, imagen




    def calculamos_error(self, centroides, shape_image):
        horizontal = shape_image[1]
        mitad = (horizontal/2) + 10
        # Pasamos los centroides a numpy para operar
        centroidesnp = np.array(centroides)
        # Cogemos la X de cada centroide
        centroidesX = centroidesnp[:,0]
        errores = centroidesX - mitad
        return (errores.sum()/len(errores))



    def calcular_pid_giro(self,kp,ki,kd,error):

        error = -error
        giro_max = 0.5
        error_max = 250
        m = giro_max/error_max
        proporcional = kp*m * error

        integral = ki*(self.integral_prev_giro +  error)
        derivada = kd * (error - self.error_prev)

        giro = (proporcional + integral + derivada)

        self.integral_prev_giro = integral


        print ('proporcional, integral, derivativo, giro final:')
        print(proporcional, integral, derivada, giro)


        return giro



    def calcular_pid_velocidad2(self,error):

        error=abs(error)

        # add a la pila de errores
        self.errores = np.append(self.errores, error)
        self.errores = np.delete(self.errores, 0)


        if all(self.errores < 11):
            return 1

        elif all(self.errores < 30):
            return 0.95

        elif all(self.errores < 50):
            return 0.9

        elif all(self.errores[10:19] < 58):
            return 0.85

        elif all(self.errores[10:19] < 65):
            return 0.8

        elif all(self.errores[5:19] < 70):
            return 0.75

        elif all(self.errores[5:19] < 75):
            return 0.7

        else:
            return 0.632 # ((proporcional + integral + derivada ))



    def setRightImageFiltered(self, image):
        self.lock.acquire()
        self.imageRight=image
        self.lock.release()


    def setLeftImageFiltered(self, image):
        self.lock.acquire()
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

