function [ u,v ] = lucasKanade(directorio1,directorio2, bloque )
%PRACTICA FLUJO OPTICO
%Parte 1. Flujo optico calculado con el método de Lucas Kanade

%esta funcion resuelve la práctica de flujo óptico con respecto a lukas
%kande para ello ese necesario pasarle a la función los directorios de
%dos imagenes y el tamaño de bloque

%La funcion va a devolver los vectores u y v además de mostrar por pantalla
%el resultado obtenido

tic

%directorio1='C:\Users\Bea\Downloads\s03\S03-COUPLE_SWAP_BAG_1\1\firstView000497.jpg';
%directorio2='C:\Users\Bea\Downloads\s03\S03-COUPLE_SWAP_BAG_1\1\firstView000505.jpg';

Im=(imread(directorio1));
Im1=(imread(directorio2));

%Im=ones(200,200,3);
%Im1=ones(200,200,3);

%Im(20:70,20:70,:)=0;
%Im1(30:80,30:80,:)=0;


%Se pasa la imagen a escala de grises
Ig=double(rgb2gray(Im));
I1g=double(rgb2gray((Im1)));

%ventana de integracion
%bloque=5
parche=(bloque-1)/2;

%Se calcula la derivada en x y en y con sobel de las dos imagenes
%Se definen las máscaras de Sobel
SobelX=[-1 0 1; -2 0 2; -1 0 1];
SobelY=[-1 -2 -1 ; 0 0 0; 1 2 1];

[x,y,z]=size(Im);
for i=bloque:(x-bloque)
    for j=bloque:(y-bloque)
         %Derivadas espaciales
        imagex=conv2(Ig(i-parche:i+parche,j-parche:j+parche),SobelX,'same');
        imagey=conv2(Ig(i-parche:i+parche,j-parche:j+parche),SobelY,'same');
        image1x=conv2(I1g(i-parche:i+parche,j-parche:j+parche),SobelX,'same');
        image1y=conv2(I1g(i-parche:i+parche,j-parche:j+parche),SobelY,'same');      

        %media del filtrado paso alto entre t y t+1
        Ix=(imagex+image1x)/2;
        Iy=(imagey+image1y)/2;
        
        %Se suavizan las imagenes
        SuavizadoIm=imgaussfilt(Ig(i-parche:i+parche,j-parche:j+parche));
        SuavizadoIm1=imgaussfilt(I1g(i-parche:i+parche,j-parche:j+parche));
        
        %A partir del suavizado se calcula It
        It=SuavizadoIm1-SuavizadoIm;
        
        %sumatorios en las vecindades locales calculando A y B 
        %[u;v] = A^-1 * B
        A=[sum(sum((Ix.^2))), sum(sum((Ix.*Iy)));sum(sum((Ix.*Iy))),sum(sum((Iy.^2)))];
        B=[-(sum(sum((Ix.*It)))) ; -(sum(sum((Iy.*It))))];   
        
        %se realiza l a pseudoinversa y se calcula u y v
        ABIm=(pinv(A))*B;   u(i,j)=-ABIm(1); v(i,j)=-ABIm(2);  
        
    end
end


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
imshow(uint8(Im1))
hold on;
% draw the velocity vectors
quiver(X_deci, Y_deci, u_deci,v_deci, 'r')

toc
end





