function [output_args]= Ej_DCT (input_args)
clc
clear

%se lee la imagen
im=imread('lena.jpg');


%Se van a crear dos operadores de DCT, uno con 1 valor a 1 y los demás a
%ceros, y otro con los 13 primeros valores de DCT a 1, y los demás a cero
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           OPERADOR DCT 1/64             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opDCT=double(zeros(8,8));
opDCT(1,1)=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           OPERADOR DCT 13/64             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opDCT2=double(zeros(8,8));
opDCT2(1,1)=1; opDCT2(3,1)=1; opDCT2(4,1)=1; opDCT2(4,2)=1;
opDCT2(1,2)=1; opDCT2(2,2)=1; opDCT2(1,4)=1; opDCT2(2,4)=1;
opDCT2(2,1)=1; opDCT2(1,3)=1; opDCT2(3,3)=1; opDCT2(3,1)=1;
                                             opDCT2(3,4)=1;
%Llamamos a las funciones que se han declarado mas abajo, 
%pasandoles la imagen que se va a filtrar y el operador  decete, en cada caso uno distinto                                    

res_1_64=ejecutarDCT(opDCT,im);
res_13_64=ejecutarDCT(opDCT2,im);

%se muestran los resultados obtenidos
figure(1)
subplot(1,2,1)
imshow(uint8(res_1_64))
title('Imagen con DCT 1/64')
subplot(1,2,2)
imshow(uint8(res_13_64))
title('Imagen con DCT 13/64')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    FUNCION QUE REALIZA LA DCT      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
function res=ejecutarDCT(operador_DCT,im)

%La funcion recibe el operador DCT, la matriz 8x8, y la imagen. devuelve
%una matriz a la cual se ha aplicado la DCT y el operador que se ha pasado.

%se calcula el tamaño de la imagen
[x,y,z]=size(im);

%y los limites de la imagen para no salinos de eela al aplicar las matrices
%8x8
limite_horizontal=x-8;
limite_vertical=y-8;

%Creamos dos bucles anidados que van a recorrer la imagen cogiendo bloques
%de 8x8

for i=1:8:limite_horizontal
 for j=1:8:limite_vertical
     
     %se halla cada bloque de la imagen y se le aplica la DCT
     matriz_11=dct2(double(im(i:i+7,j:j+7,1)));
     
     %Multiplicamos el operador DCT por el bloque 8x8 que se acaba de
     %calcular y se calcula l a inversa
     res(i:i+7,j:j+7,1)=idct2(matriz_11.*(operador_DCT));  
                
     %El resultado se guarda en una nueva matriz que es la que devuelve la
     %funcion
 end
end 

end %end function EjecutarDCT 

end %End function ej_DCT