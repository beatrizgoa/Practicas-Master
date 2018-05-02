clear
%Se cargan la iamgen derecha y la imagen izquerda, guardandose en dos
%arrays distintos.
left=(imread('b1.jpg'));
right=(imread('b2.jpg'));

%Se calcula el tampo de una de la imagen
[a,b,c]=size(right);

%Se calculan los limites de la imagen para no salirnos de ella al hacer las
%oepraciones correspondientes
limite_horizontal=a-11;
limite_vertical=b-11;

%Se declara el vector SAD donde se guardara la operacion de la suma de
%diferencias absolutas para las 50 posiciones que recorreremos(por eso el
%tamapo de sad es de 50)
sad=zeros(1,49);

%Sefinimos el tamapo del bloque que van a recorrer las imagenes
matrix_size=11;


%Se hacen dos bubles anidados para que recorren la imagen
for y=1:limite_vertical
    for x=1:limite_horizontal   
            %Se obtiene el bloque de la matriz izquierda
           matriz_izq = left(x:x+matrix_size-1, y:y+matrix_size-1);  
           i=1;
           
           %el tercer bucle se mueve  sobre la imagen de la derecha 50
           %posiciones.
        for z=x:x+49
             
            %Se hace un bloque If para que z no tome valores mayores a los
            %de las imagenes
            if z <= limite_horizontal-1
                %Se obtiene la matriz de la iamgen derecha
               matriz_dcha=right(z:z+matrix_size-1,y:y+matrix_size-1);
               
               %Se realiza la resta entre las matrices y nos quedamos con
               %el valor absoluto del resultado
               resta= abs(matriz_dcha-matriz_izq);
               
               %Sumamos los valores de la resta anterior
              sad(i)= 1/(1+(sum(sum(resta))));
               %'i' es un contador que vaaumentado conforme aumentan las 50
               %posiciones, y se reinicia 1 cuando se cambia de bloque a
               %analizar
              i=i+1;  
            end
            
        end
        
        %Se calcula el valor minimo del vector sad
        MIN=min(sad);
       % Y buscamos la posicion donde esta ese minimo  
         Posiciones=find(sad==MIN);
       %Guardamos el valor de la posicion en una nueva matriz que
         mapa(x,y)=Posiciones(1,1);
       
    end
    
end

%Se pasa el mapa a integers y se normaliza enre un espacio de 0 a 255 ya
%que los valores se encuentran entre 0 y 255

mapa_disparidad=uint8(mapa);

%Mostramos el resutado obtenido
imshow(mapa_disparidad,[])
