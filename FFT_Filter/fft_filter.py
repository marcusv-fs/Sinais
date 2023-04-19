import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt 
import os 
import glob #utilizado para encontrar arquivos especificos (nesse caso sao arquivos .xml)
import xml.etree.ElementTree as ET 

def filtro_1(img, r): 
#LOW PASS FILTER
  #-------------------------------------------------------------------------------------
  dft = np.fft.fft2(img[:, :, 2])  #aplicando a dft no canal RED da imagem
  dftShift = np.fft.fftshift(dft) #deslocando o elemento de w = 0 pro centro

  lowPass1 = np.hamming(img.shape[0])[:,None] #1D hamming
  lowPass2 = np.hamming(img.shape[1])[:,None] 
  lowPass2d = np.sqrt(np.outer(lowPass1, lowPass2)) ** r 

  dftShift =  lowPass2d * dftShift

  inv_dftShift = np.fft.ifftshift(dftShift) 
  inv_dft = abs(np.fft.ifft2(dftShift))
  inv_dft_red = inv_dft.clip(0,255) 
  #-------------------------------------------------------------------------------------
  dft = np.fft.fft2(img[:, :, 1])  #aplicando a dft no canal GREEN da imagem
  dftShift = np.fft.fftshift(dft) #deslocando o elemento de w = 0 pro centro

  dftShift =  lowPass2d * dftShift

  inv_dftShift = np.fft.ifftshift(dftShift) 
  inv_dft = abs(np.fft.ifft2(dftShift))
  inv_dft_green = inv_dft.clip(0,255) 
  #-------------------------------------------------------------------------------------
  dft = np.fft.fft2(img[:, :, 0])  #aplicando a dft no canal BLUE da imagem
  dftShift = np.fft.fftshift(dft) #deslocando o elemento de w = 0 pro centro

  dftShift =  lowPass2d * dftShift

  inv_dftShift = np.fft.ifftshift(dftShift) 
  inv_dft = abs(np.fft.ifft2(dftShift)) 
  inv_dft_blue = inv_dft.clip(0,255) 
  #-------------------------------------------------------------------------------------
  nova_img = np.dstack([inv_dft_red.astype(int), inv_dft_green.astype(int), inv_dft_blue.astype(int)]) #juntando os tres canais

  return nova_img

def filtro_2(img): 
#NOISE FILTER
  #-------------------------------------------------------------------------------------
  dft = np.fft.fft2(img[:, :, 2])  #aplicando a dft no canal RED da imagem
  dftShift = np.fft.fftshift(dft) #deslocando o elemento de w = 0 pro centro

  noise = np.random.rand(*dftShift.shape) #criando uma imagem de ruido aleatorio
  dftShift *= noise 

  inv_dftShift = np.fft.ifftshift(dftShift) 
  inv_dft = abs(np.fft.ifft2(dftShift)) 
  inv_dft_red = inv_dft.clip(0,255)  
  #-------------------------------------------------------------------------------------
  dft = np.fft.fft2(img[:, :, 1])  #aplicando a dft no canal GREEN da imagem
  dftShift = np.fft.fftshift(dft) #deslocando o elemento de w = 0 pro centro

  noise = np.random.rand(*dftShift.shape) #criando uma imagem de ruido aleatorio
  dftShift *= noise 

  inv_dftShift = np.fft.ifftshift(dftShift) 
  inv_dft = abs(np.fft.ifft2(dftShift)) 
  inv_dft_green = inv_dft.clip(0,255)  
  #-------------------------------------------------------------------------------------
  dft = np.fft.fft2(img[:, :, 0])  #aplicando a dft no canal BLUE da imagem
  dftShift = np.fft.fftshift(dft) #deslocando o elemento de w = 0 pro centro
  
  noise = np.random.rand(*dftShift.shape) #criando uma imagem de ruido aleatorio
  dftShift *= noise 

  inv_dftShift = np.fft.ifftshift(dftShift) 
  inv_dft = abs(np.fft.ifft2(dftShift)) 
  inv_dft_blue = inv_dft.clip(0,255) 
  #-------------------------------------------------------------------------------------
  nova_img = np.dstack([inv_dft_red.astype(int), inv_dft_green.astype(int), inv_dft_blue.astype(int)]) #juntando os tres canais

  return nova_img

lista_de_imagens = os.listdir('/content/drive/MyDrive/VOC2012/JPEGImages') # <--DIRETORIO DE IMAGENS AQUI
lista_de_imagens.sort()

# Aplicando o filtro 1 na primeira metade das imagens originais
for i in lista_de_imagens[0:int(len(lista_de_imagens)/2)]: 
  img = cv.imread('/content/drive/MyDrive/VOC2012/JPEGImages/'+i, cv.IMREAD_COLOR) 
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  
  xml_da_img = '/content/drive/MyDrive/VOC2012/bboxes/annotations/'+i[0:10]+'.xml' 
  if os.path.exists(xml_da_img): #Algumas imagens nao possuem xml
    tree = ET.parse(xml_da_img) 
    root = tree.getroot() 
    nome_antigo = root.find('filename')
    nome_antigo.text = nome_antigo.text.replace(i, i[0:10]+'_FFT'+'.jpg') 
    caminho_antigo = root.find('path')
    caminho_antigo.text = caminho_antigo.text.replace(i, i[0:10]+'_FFT'+'.jpg') 
    tree.write('/content/xml_imagens_filtro_1/'+i[0:10]+'_FFT'+'.xml')

    # Analise do tamanho da imagem (Caso muito pequena, o filtro e reduzido)
    area_pequena = False
    for objeto in root.iter('object'):
      area = (int(objeto.find('bndbox/xmax').text)-int(objeto.find('bndbox/xmin').text))*(int(objeto.find('bndbox/ymax').text)-int(objeto.find('bndbox/ymin').text)) #area da bounding box
      if area < 4000:
          area_pequena = True 
          break
    if area_pequena: 
      nova_img = filtro_1(img, 5) 
    else:
      nova_img = filtro_1(img, 50) 

    cv.imwrite('/content/imagens_filtro_1/'+i[0:10]+'_FFT'+'.jpg', nova_img) # NomeAntigo_FFT.jpg

# Aplicando o filtro 2 na segunda metade das imagens originais
for i in lista_de_imagens[int(len(lista_de_imagens)/2):]: 
  img = cv.imread('/content/drive/MyDrive/VOC2012/JPEGImages/'+i, cv.IMREAD_COLOR) 
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  nova_img = filtro_2(img)

  cv.imwrite('/content/imagens_filtro_2/'+i[0:10]+'_FFT'+'.jpg', nova_img) # NomeAntigo_FFT.jpg
  xml_da_img = '/content/drive/MyDrive/VOC2012/bboxes/annotations/'+i[0:10]+'.xml' 
  if os.path.exists(xml_da_img): 
    tree = ET.parse(xml_da_img) 
    root = tree.getroot() 
    nome_antigo = root.find('filename')
    nome_antigo.text = nome_antigo.text.replace(i, i[0:10]+'_FFT'+'.jpg') 
    caminho_antigo = root.find('path')
    caminho_antigo.text = caminho_antigo.text.replace(i, i[0:10]+'_FFT'+'.jpg') 
    tree.write('/content/xml_imagens_filtro_2/'+i[0:10]+'_FFT'+'.xml')
