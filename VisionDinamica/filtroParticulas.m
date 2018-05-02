%En este script/funcion se va a desarrollar el filtro de particulas.
ruta_entrada='SecuenciaPelota\';
pictures_jpg = [ruta_entrada '*.jpg'];
lee_archivos = dir(pictures_jpg); %directorio de donde leemos.

for k = 1:length(lee_archivos) %recorre número de archivos guardados en el directorio
    
    ima = lee_archivos(k).name;
    ima = imread(strcat(ruta_entrada,ima));% lee la  imagen
    imagen=rgb2gray(ima);
    
    if k==1
       im0=imagen;
       continue
    end 
   
    if k==2
        %Se definen variables:
        N = 1000; %numero de particulas

        %%0. Se extrae de la imagen el  fondo
        resultado = imagen-im0;
        
        [tamx,tamy,tamz] = size(imagen);
        %%1. Iniciacion de tipo aleatoria por la imgegen
        a = 1;
        b1 = tamx;
        b2 = tamy;

        x_val = (b1-a).*rand(N,1,'single') + a;
        y_val = (b2-a).*rand(N,1,'single') + a;

        %Los píxeles de la imagen
        pixeles = [x_val y_val];
        pixeles = round(pixeles);

        %%2. Evaluacion del peso de las particuas

        tam_bloque = 18; %El bloque va a ser de 37x37 (aprox el tamaño del objeto)

        %valormax=255*37*37;
        valormax=300000; %es un valor algo menor a 255*37*37 
        resultado_pad= padarray(resultado,[tam_bloque tam_bloque]);

        for i=1:N
            valorx = pixeles(i,1);
            valory = pixeles(i,2);
            bloque = resultado_pad(valorx:valorx+2*tam_bloque,valory:valory+2*tam_bloque);
            pixeles(i,3)=sum(sum(bloque))/valormax; %en esta se guarda el valor acumulado normalizado, que va a ser el peso (el valor va entre 0 y 1)
        end

        %%3. Estimación. NOs quedamos con la particula de mayor peso

        particula=[0,0,0];
        
        for i=1:length(pixeles)
            if pixeles(i,3)>particula(3)
                particula=pixeles(i,:);
            end
        end

        
        figure(1)
        imshow(ima);
        hold on;
        rectangle('Position',[particula(2)-20, particula(1)-20, 40,40],'EdgeColor','r')
        %waitforbuttonpress()
        pause(0.8)

    else
        %%4. Selección
        %A partir de la particula ganadora se hará una difusión de 15 particulas en
        %el vecindario de 51x51 al rededor de la particula.
        
        resultado = imagen-im0;
        
        tam_bloque2=50;
        resultado_pad2= padarray(resultado,[tam_bloque2 tam_bloque2]);  %La imagen con ceros 
        particula = particula + tam_bloque2;

        %%5. Difusion
        N2=100;
            a1 = particula(1)-tam_bloque2;  %Hacemos una nueva difusion en la imagen con ceros con vecindario tambloque2xtambloque2
            a2 = particula(2)-tam_bloque2;
            b1 = particula(1)+tam_bloque2;
            b2 = particula(2)+tam_bloque2;

            x_val2 = (b1-a1).*rand(N2,1,'single') + a1;
            y_val2 = (b2-a2).*rand(N2,1,'single') + a2;

            pixeles2 = [x_val2 y_val2];
            pixeles2 = round(pixeles2); %Los valores de estos pixeles estan en la imagen con cero
           
            for i=1:N2
                valorx2 = pixeles2(i,1);
                valory2 = pixeles2(i,2);
                bloque2 = resultado_pad2(valorx2-tam_bloque:valorx2+tam_bloque,valory2-tam_bloque:valory2+tam_bloque); %Se coje la vecindad tambloquextambloque del pixel en la im con ceros
                pixeles2(i,3)=sum(sum(bloque2))/valormax; %en esta se guarda el valor acumulado normalizado (el valor va entre 0 y 1)
            end

            particula2=[0,0,0];

            for i=1:N2
                if pixeles2(i,3) > particula2(3)
                    particula2=pixeles2(i,:); %nos quedamos con el de mayor peso
                end
            end

            
        nueva_particula2=[particula2(1)-tam_bloque2, particula2(2)-tam_bloque2]; %Le quitamos el tamaño del bloque 2 a l a nuevaparticula 
        figure(1)
        imshow(ima);
        hold on;
        % Then, from the help:
        rectangle('Position',[nueva_particula2(2)-20, nueva_particula2(1)-20, 40,40],'EdgeColor','r')
        %waitforbuttonpress()
        pause(0.8)

        particula=nueva_particula2;
    end
end

