"""
Herramientas Software: Ejercicio 1 de python.
Alumna: Beatriz Gomez Ayllon
"""

# SE CARGAN LAS LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt


# Funcion para preparar las graficas
def preparamos_graficamos(p_area2d, p_area3d, complexity):

    # Preparamos la grafica del area2D
    #PARAMETROS PARA EL GRAFICO
    ind = np.arange(6)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    # Preparamos la grafica de area2d
    figure2d = plt.figure('graf_area2D')
    plt.bar(ind,p_area2d,width,color='g')
    plt.ylabel('%')
    plt.title('Area 2D')
    plt.xticks(ind+width/2., ('Error', '[0-50)', '[50-100)', '[100-150)', '[150-200)' , 'mayor 200') )

    # Preparamos la grafica de Area3d
    figure3d = plt.figure('graf_Area3D')
    plt.bar(ind,p_area3d,width,color='g')
    plt.ylabel('%')
    plt.title('Area 3D')
    plt.xticks(ind+width/2., ('Error', '[0-50)', '[50-100)', '[100-150)', '[150-200)' , 'mayor 200') )

    # Preparamos la grafica de la complejidad
    figure_complex = plt.figure('graf_Complexity')
    plt.bar(ind,complexity,width,color='g')
    plt.ylabel('Count')
    plt.title('Complexity')
    plt.xticks(ind+width/2., ('Error', 'Dif.=0', 'Dif.=1', 'Dif.=2', 'Dif.=3' , 'Dif.>=4') )

    return [figure2d, figure3d, figure_complex]

# Funcion para guardar una grafica
def guardar_graficas(figures, save_path):
    figures[0].savefig(save_path + "\\" + 'area2d.jpg')
    figures[1].savefig(save_path+"\\"+'area3d.jpg')
    figures[2].savefig(save_path+"\\"+'complexity.jpg')


def compute_stats(csv1_path, csv2_path, save_path):
    #COJEMOS LOS CSV
    detect = np.genfromtxt(csv1_path, delimiter="," , skiprows=1,missing='-')
    ground = np.genfromtxt(csv2_path, delimiter="," , skiprows=1,missing='-')

    #################################
    #Se cuentan los errores: El error es general para los tres valores, por eso solo cojemos los de una columnas
    error = 0
    for fila in detect:
        for valor in fila:
            if np.isnan(valor):
                error += 1


    #Guarmamos el tamaNYo del archivo detection en tam
    tam = detect.shape

    ####################################3333
    #Se calculan las similitudes para los tres valores
    count=np.equal(detect, ground)

    ####################################33

    #Se suma por separado los intervalos.
    #Se crearan distintos intervalos, en area 2d y area 3d  corresponden con intervalos de id.
    #La complejidad se mostrara segun la diferencia de valor entre los dos archivos csv de cada id.

    #area 2d
    Count_Area2d_50 = np.sum(count[0:50,1])
    Count_Area2d_100 = np.sum(count[50:100,1])
    Count_Area2d_150 = np.sum(count[100:150,1])
    Count_Area2d_200 = np.sum(count[150:200,1])
    Count_Area2d_mas = np.sum(count[200:tam[0],1])

    #area 3d
    Count_Area3d_50 = np.sum(count[0:50,2])
    Count_Area3d_100 = np.sum(count[50:100,2])
    Count_Area3d_150 = np.sum(count[100:150,2])
    Count_Area3d_200 = np.sum(count[150:200,2])
    Count_Area3d_mas = np.sum(count[200:tam[0],2])

    #Diferenicas de la complejidad
    Complexity_0=0
    Complexity_1=0
    Complexity_2=0
    Complexity_3=0
    Complexity_mas=0
#aumentamos un valor en cada contador, segun la dif en valor absoluto de la celda correspondiente de detection con groundtruth
    for index, value in enumerate(detect):
        if value[3]==ground[index,3]:
            Complexity_0+=1 #Cuando son iguales
        elif abs(value[3]-ground[index,3])==1:
            Complexity_1+=1 #Cuando la diferencia es 1
        elif abs(value[3]-ground[index,3])==2:
            Complexity_2+=1 #Cuando la diferencia es 2
        elif abs(value[3]-ground[index,3])==3:
            Complexity_3+=1 #Cuando la diferencia es 3
        else:
            Complexity_mas+=1 #Para los demas casos, cuando es cuatro o mayor la dif

    ###############################
    #Se calcula la probabilidad de error
    P_error=100*error/tam[0]
    print 'El error es (%):'
    print P_error


    #Calculamos los porcentajes y los guardamos en un array que sera utilizado para la representacion
    P_Area2D = ([P_error,100*Count_Area2d_50/tam[0],100*Count_Area2d_100/tam[0],100*Count_Area2d_150/tam[0],100*Count_Area2d_200/tam[0],100*Count_Area2d_mas/tam[0]])
    P_Area3D = np.array([P_error,100*Count_Area3d_50/tam[0],100*Count_Area3d_100/tam[0],100*Count_Area3d_150/tam[0],100*Count_Area3d_200/tam[0],100*Count_Area3d_mas/tam[0]])
    Complexity = np.array([error,Complexity_0,Complexity_1,Complexity_2,Complexity_3,Complexity_mas])

    #Imprimimos valores de porcentajes
    print 'Porcentaje area 2d'
    print P_Area2D

    print 'Porcentaje area 3d'
    print P_Area3D

    print 'Diferencia Complexity'
    print Complexity

    # Preparamos las tres graficos, para posteriormente ser guardadas o imprimidas
    figuras = preparamos_graficamos(P_Area2D, P_Area3D, Complexity)

    # Guardamos las graficas
    guardar_graficas(figuras, save_path)

    # Mostramos los tres graficos :)
    plt.show()


#AQUI SE LLAMA A LA FUNCION REQUERIDA
compute_stats('detection.csv','groundtruth.csv','C:\Users\Bea\Desktop\master\herramientas software\python_workspace\practica1')
