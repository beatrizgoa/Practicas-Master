%PRACTICA FLUJO OPTICO
%Parte 2. Flujo optico calculado con el método de Horn y Schunck
function [ u,v ] = HornSchunck(directorio1,directorio2, lamda, convergencia )

%esta funcion resuelve la práctica de flujo óptico con respecto a Horn y Schunck, para ello ese necesario pasarle a la función los directorios de
%dos imagenes, el valor de lamba y elnumero de iteraciones hasta lllegar a
%convergencia

%La funcion va a devolver los vectores u y v además de mostrar por pantalla
%el resultado obtenido

tic

%Im=ones(200,200,3);
%Im1=ones(200,200,3);

%Im(20:70,20:70,:)=0;
%Im1(30:80,30:80,:)=0;

%directorio1='C:\Users\Bea\Downloads\s03\S03-COUPLE_SWAP_BAG_1\1\firstView000497.jpg';
%directorio2='C:\Users\Bea\Downloads\s03\S03-COUPLE_SWAP_BAG_1\1\firstView000505.jpg';

Im=(imread(directorio1));
Im1=(imread(directorio2));

%Se pasa la imagen a escala de grises
Ig=double(rgb2gray(Im));
I1g=double(rgb2gray((Im1)));

%Se suavizan las imagenes
Ig=imgaussfilt(Ig,1);
I1g=imgaussfilt(I1g,1);

%Se inicializan las variables globales
%lamda=40
%convergencia=10

[x,y,z]=size(Ig);

%se inicializan los valores u y v medios anteriores
Ukmed_ant=0;
Vkmed_ant=0;

%Se calculan las derivadas y promedio espacio temporales
Ix = conv2(Ig,[1 -1]);
Ix=Ix(1:x,1:y);
Iy = conv2(I1g,[1; -1]);
Iy=Iy(1:x,1:y);

%A partir del suavizado se calcula It
It= Ig-I1g;

%Se define la mascara que viene dado en el paper original
Mascara=[1 2 1; 2 -12 2; 1 2 1]/12;

%Se repiten las ecuaciones iterativas hasta convergencia o num max de
%iteraciones
for a=1:convergencia 
%Se calcula el promedio en una vecindad pequeña
Ukmed=conv2(Ukmed_ant,Mascara,'same');
Vkmed=conv2(Vkmed_ant,Mascara,'same');

%Se calcula las ecuaciones iterativas
uk=Ukmed_ant - (Ix .* (Ix * Ukmed_ant + Iy * Vkmed_ant + It) ./ (lamda^2 + Ix.^2 + Iy.^2));
vk=Vkmed_ant - (Iy .* (Ix * Ukmed_ant + Iy * Vkmed_ant + It) ./(lamda.^2 + Ix.^2 + Iy.^2));

%se pasa el valor de u/v med al anterior
Ukmed_ant=Ukmed;   Vkmed_ant=Vkmed;      
end

%se obtienen u y v
u=-uk; v=-vk;

%Para mostrar la imagen:
% downsize u and v
u_deci = u(1:10:end, 1:10:end);
v_deci = v(1:10:end, 1:10:end);
% get coordinate for u and v in the original frame
[m, n] = size(u);
[X,Y] = meshgrid(1:n, 1:m);
X_deci = X(1:10:end, 1:10:end);
Y_deci = Y(1:10:end, 1:10:end);

%Se muestra
figure(1);
imshow((Im1))
%imshow(uint8(Im1))
hold on;
% draw the velocity vectors
quiver(X_deci, Y_deci, u_deci,v_deci, 'r')

toc

end
    