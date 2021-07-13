from os import DirEntry
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import array
import random

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def normalize(vector):
    return vector / np.linalg.norm(vector)

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin -center) **2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) /2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1,t2)
    return None

def nearest_intersect_objects(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


objects = [
    {'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1,0,0]), 'diffuse': np.array([0.7,0,0]), 'specular':np.array([1,1,1]), 'shininess':10000, 'reflection': 0.5},
    {'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1,0,0.1]), 'diffuse': np.array([0.7,0,0.7]), 'specular':np.array([1,1,1]), 'shininess':10000, 'reflection': 0.5},
    {'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0,0.1,0]), 'diffuse': np.array([0,0.6,0]), 'specular':np.array([1,1,1]), 'shininess':10000, 'reflection': 0.5},
    {'center': np.array([0, -9000, 0]), 'radius': 9000-0.7, 'ambient': np.array([0.1,0.1,0.1]), 'diffuse': np.array([0.6,0.6,0.6]), 'specular':np.array([1,1,1]), 'shininess':10000, 'reflection': 0.5}
]

lights = [
    {'center': np.array([5,5,5]), 'radius':0.7,'ambient': np.array([1,1,1]), 'diffuse': np.array([1,1,1]), 'specular':np.array([1,1,1])}
]


width = 300
height = 200

camera = np.array([0, 0, 1])
ratio = float(width)/ height
screen = (-1, 1 / ratio, 1, -1 / ratio)

image = np.zeros((height, width, 3))
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        pixel = np.array([x,y,0])
        origin = camera
        direction = normalize(pixel - origin)

        nearest_object, min_distance = nearest_intersect_objects(objects, origin,direction)
        if nearest_object is None:
            continue

        intersection = origin + min_distance * direction

        normal_to_surface = normalize(intersection - nearest_object['center'])
        shifted_point = intersection + 1e-5 * normal_to_surface


        illumination = np.zeros((3))

        sample = 1

        color = np.zeros((3))
        reflection = 1
        for l in lights:

            light_center = l['center']

            for c in range(0,100):

                for k in range(2):
                    r = random.random()
                    s = random.random()
                    #for c in range(-2,3):
                    #   for v in range(-2,3):
                            #for b in range(-1,2):

                    light_pos = [light_center[0],light_center[1]+r*5,light_center[2]+s*5]
                    
                    instersection_to_light = normalize(light_pos - shifted_point)

                    _, min_distance = nearest_intersect_objects(objects, shifted_point, instersection_to_light)
                    intersection_to_light_distance = np.linalg.norm(light_pos - intersection)
                    is_shadowed = min_distance < intersection_to_light_distance

                    if is_shadowed:
                        continue



                    illumination += nearest_object['ambient'] * l['ambient']
                    illumination += nearest_object['diffuse'] * l['diffuse'] * np.dot(instersection_to_light, normal_to_surface)
                    sample+=1
                    intersection_to_camera = normalize(camera - intersection)
                    H = normalize(instersection_to_light + intersection_to_camera)
                    illumination += nearest_object['specular'] * lights[0]['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)
                    
                    color += reflection * illumination
                    reflection *= nearest_object['reflection']
                    origin = shifted_point
                    direction = reflected(direction, normal_to_surface)

        image[i,j] = np.clip(color/2,0,1)
        print("progress: %d/%d" % (i + 1, height))

plt.imsave('image.png', image)
