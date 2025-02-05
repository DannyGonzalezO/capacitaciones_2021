#!/usr/bin/env python
# prueba
"""
Este programa permite mover al Duckiebot dentro del simulador
usando el teclado.
"""

import sys
import argparse
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import cv2

# Se leen los argumentos de entrada
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Duckietown-udem1-v1")
parser.add_argument('--map-name', default='mapaDGonzalez')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# Definición del environment
if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

# Se reinicia el environment
env.reset()

old_key=-1
while True:

    # Captura la tecla que está siendo apretada y almacena su valor en key
    key = cv2.waitKey(30)

    # Si la tecla es Esc, se sale del loop y termina el programa
    if key == 27:
        break

    # La acción de Duckiebot consiste en dos valores:
    # velocidad lineal y velocidad de giro
    # En este caso, ambas velocidades son 0 (acción por defecto)
    action = np.array([0.0, 0.0])

    # Definir acción en base a la tecla apretada

    # Esto es avanzar recto hacia adelante al apretar la tecla w
    if key == ord('w'):
        action += np.array([0.5, 0.0])
#giro izquierda
    if key == ord('a'):
        action += np.array([0, 1.0])
#giro derecha
    if key == ord('d'):
        action += np.array([0, -1.0])
#retroceder
    if key == ord('s'):
        action += np.array([-0.5, 0.0])
#diagonal adelante izquierda:
    if key == ord('q'):
        action += np.array([0.5, 1.0])
#diagonal adelant derecha:
    if key == ord('e'):
        action += np.array([0.5, -1.0])
#turbo adelante (mas rapido)
    if key == ord('p'):
        action += np.array([1.5, 0.0])

# borrador, no aplica nada todavía        
##    if action[0]>(0):
##        if old_key==ord('w') and key == (-1):
##            action += np.array([0.5, 0.0])
##        if old_key==(-1):
##            action += action - np.array([0.05, 0.0])
##
####    if action[0]<(0.5):
####            action += action + np.array([+0.08, 0.0])
                

    ### AGREGAR MÁS COMPORTAMIENTOS ###
    old_key = key


    # Se ejecuta la acción definida anteriormente y se retorna la observación (obs),
    # la evaluación (reward), etc
    obs, reward, done, info = env.step(action)
    # obs consiste en un imagen de 640 x 480 x 3

    # done significa que el Duckiebot chocó con un objeto o se salió del camino
    if done:
        print('done!')
        # En ese caso se reinicia el simulador
        env.reset()

    # Se muestra en una ventana llamada "patos" la observación del simulador
    cv2.imshow("MCF", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))


# Se cierra el environment y termina el programa
env.close()
