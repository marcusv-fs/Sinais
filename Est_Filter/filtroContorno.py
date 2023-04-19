from PIL import Image, ImageFilter
from math import sqrt
from shutil import copyfile
import os
import numpy as np

# Diretório onde as imagens e seus respectivos XML estão e serão armazenados
imageDir = os.path.join('VOC2012', 'JPEGImages')
annotationsDir = os.path.join('VOC2012', 'bboxes', 'annotations')
#print(annotationsDir)

def copyXML(filename):
    if(os.path.exists(annotationsDir+filename+'.xml')):
        copyfile(annotationsDir+filename+'.xml',annotationsDir+filename+'_FE.xml')

def showEdges(filename, offset=0):
    #Aplica um filtro Sobel à imagem, exibe e salva o resultado

    original = Image.open(imageDir+filename).convert('L')
    XSOBEL = ImageFilter.Kernel((3, 3),
                                [
                                -1, 0, 1,
                                -2, 0, 2,
                                -1, 0, 1
                                ],
                                1, 
                                offset)
    YSOBEL = ImageFilter.Kernel((3, 3),
                                [
                                -1, -2, -1,
                                 0,  0,  0,
                                 1,  2,  1
                                ],
                                1,
                                offset)
    
    vsobel = original.filter(XSOBEL)
    hsobel = original.filter(YSOBEL)
    w, h = original.size
    filtered = Image.new('L', (w, h))

    for i in range(w):
        for j in range(h):
            value = sqrt(
                vsobel.getpixel((i, j))**2 + hsobel.getpixel((i, j))**2
            )
            value = int(min(value, 255))
            filtered.putpixel((i, j), value)
    
    #FE significa aqui 'Filtro Estatístico'
    filtered.save(imageDir+'{}_FE.jpg'.format(filename[:filename.index('.')]))

nameFile = '\image_'
for i in range(1686): #número de imagens no nosso banco de dados reservadas para teste
    if(i<10):
        number = '000'+str(i)
    elif(i<100):
        number = '00'+str(i)
    elif(i<1000):
        number = '0'+str(i)
    else:
        number = str(i)
    showEdges(nameFile+number+'.jpg', 0)
    copyXML(nameFile+number)