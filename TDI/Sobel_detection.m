function [ output_args ] = Sobel_detection( input_args )
clc 
clear
close all

%ESTA FUNCION RESUELVE EL EJERCICIO 1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DOMINIO DE LA IMAGEN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Cargamos la imagen y la transformamos a escala de grises
%image=imread('sobel.jpg'); %IMAGEN DE TEXTO
image=imread('lena.jpg'); %IMAGEN DE LENA


%Se pasa la imagen a escala de grises
image=rgb2gray(image);
image=double(image);

%Se definen las máscaras de Sobel
GX=[-1 0 1; -2 0 2; -1 0 1];
GY=[-1 -2 -1 ; 0 0 0; 1 2 1];

%Se convoluciona  la máscara con la imagen para ambas direcciones x e y
imagex=conv2(GX,image);
imagey=conv2(GY,image);

%Se calcula el gradiente 
im_sobel1= sqrt(imagex.^2+ imagey .^2) ;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DOMINIO DE LA FRECUENCIA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Pasamos al dominio de la frecuencia la imagen y calculamos su tamaño
fimage=(fft2(double(image)));
fimage=fftshift(fimage);
%fimage=double(fimage);
[x,y]=size(fimage);

%se crea el filtro 
h = fspecial('sobel');
%Hacemos dosfiltros uno para calcular los bordes horizontales y otros
%verticales

F_horizontal=imfilter(fimage,h);
F_vertical=imfilter(fimage,h.');


%Se pasan al dominio frecuencial los fltros:
F_horizontal=fftshift(fft2(F_horizontal));
F_vertical=fftshift(fft2(F_vertical));


%Se calculan los bordes para ambas direcciones aplicando la imagen al
%filtro
res1=fimage.*F_horizontal;
res2=fimage.*F_vertical;

%Cojo el modulo despues de realizar las  inversas
res1=(abs(ifftshift(ifft2(res1))));
res2=(abs(ifftshift(ifft2(res2))));

%Sumo el resultado de la obtencion de amos bordes
im_sobel2=res1+res2;


%se muestran las dos imagenes para compararlas en el dominio de la imagen y
%en el dominio de la frecuencia

figure(1)
subplot(2,2,1)
imshow(im_sobel1,[])
title('Domino de la Imagen')

subplot(2,2,2)
imshow(im_sobel2,[])
title('Dominio de la Frecuencia')


%Borde shorizontales y  verticales en el domnio de la frecuenica
subplot(2,2,3)
imshow(res1,[])
title('bordes horizontales en frec')

subplot(2,2,4)
imshow(res2,[])
title('Bordes verticales en frec')


end

