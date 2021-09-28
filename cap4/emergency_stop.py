#!/usr/bin/env python

"""
Este programa implementa un freno de emergencia para evitar accidentes en Duckietown.
"""

import sys
import argparse
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import cv2

def mov_duckiebot(key):
    # La acción de Duckiebot consiste en dos valores:
    # velocidad lineal y velocidad de giro
    actions = {ord('w'): np.array([1.0, 0.0]),
               ord('s'): np.array([-1.0, 0.0]),
               ord('a'): np.array([0.0, 1.0]),
               ord('d'): np.array([0.0, -1.0]),
               ord('q'): np.array([0.3, 1.0]),
               ord('e'): np.array([0.3, -1.0])
               }

    action = actions.get(key, np.array([0.0, 0.0]))
    return action


def det_duckie(obs):
    B_m=0
    G_m=230 
    R_m=175 
    B_M=100 
    G_M=255
    R_M=255 

    lower_yellow = np.array([B_m, G_m, R_m])
    upper_yellow = np.array([B_M, G_M, R_M])
    min_area = 800 
    ### DETECTOR HECHO EN LA MISIÓN ANTERIOR
    dets = list()
    #Transformar imagen a espacio HSV
    img_outHSV = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    # Filtrar colores de la imagen en el rango utilizando
    ## img_in por img_outHSV
    mask = cv2.inRange(img_outHSV, lower_yellow, upper_yellow)
    ## img_in por img_outHSV
    # Bitwise-AND entre máscara (mask) y original (obs) para visualizar lo filtrado
    img_out = cv2.bitwise_and(img_outHSV, img_outHSV, mask = mask)

    # Se define kernel para operaciones morfológicas
    kernel = np.ones((5,5),np.uint8)

    # Aplicar operaciones morfológicas para eliminar ruido
    # Esto corresponde a hacer un Opening
    # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    #Operacion morfologica erode
    mask_erode = cv2.erode(mask, kernel, iterations = 1)

    #Operacion morfologica dilate
    mask_dilate = cv2.dilate(mask_erode, kernel, iterations = 1)

    # Busca contornos de blobs
    # https://docs.opencv.org/trunk/d3/d05/tutorial_py_table_of_contents_contours.html
    contours, hierarchy = cv2.findContours(mask_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        # Obtener rectangulo que bordea un contorno
        ## coordenadas, ancho y alto(?
        x, y, w, h = cv2.boundingRect(cnt)
        print (x, y)

        # DEFINIR AREA
        #AREA = abs(x-y)*abs(w-h)
        AREA = abs(w*h)
        
        

        if AREA > min_area:
            # En lugar de dibujar, se agrega a la lista
            dets.append((x,y,w,h))

    return dets

def draw_dets(obs, dets):
    for d in dets:
        x1, y1 = d[0], d[1]
        x2 = x1 + d[2]
        y2 = y1 + d[3]
        cv2.rectangle(obs, (int(x1), int(y1)), (int(x2),int(y2)), (0,255,0), 3)

    return obs

def red_alert(obs):
    red_img = np.zeros((480, 640, 3), dtype = np.uint8)
    red_img[:,:,0] = 255
    blend = cv2.addWeighted(obs, 0.5, red_img, 0.5, 0)

    return blend

if __name__ == '__main__':

    # Se leen los argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="Duckietown-udem1-v1")
    parser.add_argument('--map-name', default='udem1')
    args = parser.parse_args()

    # Definición del environment
    if args.env_name and args.env_name.find('Duckietown') != -1:
        env = DuckietownEnv(
            map_name = args.map_name,
            domain_rand = False,
        )
    else:
        env = gym.make(args.env_name)

    # Se reinicia el environment
    env.reset()

    # Inicialmente no hay alerta
    alert = False

    # Posición del pato en el mapa (fija)
    duck_pos = np.array([2,0,2])

    # Constante que se debe calcular
##    C = 875*0.08 # f * hr (f es constante, hr es conocido) (a distancias cortas funcionaba bien, pero cuando está bajo los dos centimetros, estima una distancia mayor a la real, por lo que se descarta)
    C = 750*0.08 # corresponde a lo obtenido experimentalente, funciona bien a distancias cercanas a 1 metro, pero mientras está cerca tiene un error de aprox 2-4 cm, aunque suele estimar una distancia menor, por lo que no sería tan peligroso)

    while True:

        # Captura la tecla que está siendo apretada y almacena su valor en key
        key = cv2.waitKey(0)
        # Si la tecla es Esc, se sale del loop y termina el programa
        if key == 27:
            break

        # Se define la acción dada la tecla presionada
        action = mov_duckiebot(key)

        # Si hay alerta evitar que el Duckiebot avance (lo frena o retrocede)
        #este le disminuye la velocidad, pero sigue avanzando
##        if alert:
##            if key == ord('w'):
##                action = np.array([0.1, 0.0])
##            pass
        #este si lo frena
        if alert:
            if key == ord('w'):
                action = np.array([0.0, 0.0])
            pass

        # Se ejecuta la acción definida anteriormente y se retorna la observación (obs),
        # la evaluación (reward), etc
        obs, reward, done, info = env.step(action)

        # Detección de patos, retorna lista de detecciones
        DetPatos = det_duckie(obs) #

        # Dibuja las detecciones
        obs = draw_dets(obs, DetPatos) #

        # Obtener posición del duckiebot
        dbot_pos = env.cur_pos
        # Calcular distancia real entre posición del duckiebot y pato
        # esta distancia se utiliza para calcular la constante
        dist = ((duck_pos[0]-dbot_pos[0])**2+(duck_pos[1]-dbot_pos[1])**2 +(duck_pos[2]-dbot_pos[2])**2)**0.5

        # La alerta se desactiva (opción por defecto)
        alert = False
        
        for d in DetPatos:
            # Alto de la detección en pixeles
            p = d[3]
            # La aproximación se calcula según la fórmula mostrada en la capacitación
            d_aprox = C/p

            # Muestra información relevante
            print('p:', p)
            print('Da:', d_aprox)
            print('Dr:', dist)

            # Si la distancia es muy pequeña activa alerta
            if d_aprox < 0.3:
                # Activar alarma
                alert = True

                # Muestra ventana en rojo
                obs=red_alert(obs)

        # Se muestra en una ventana llamada "patos" la observación del simulador
        cv2.imshow('patos', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    # Se cierra el environment y termina el programa
    env.close()
