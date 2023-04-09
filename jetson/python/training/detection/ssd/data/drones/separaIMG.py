import os
import shutil


path_origem = 'test'
path_destino = 'JPEGImages'


for item in [os.path.join(path_origem, f) for f in os.listdir(path_origem) if os.path.isfile(os.path.join(path_origem, f)) and f.endswith('jpg')]:
        #print(item)
        shutil.move(item, os.path.join(path_destino, os.path.basename(item)))
        print('moved "{}" -> "{}"'.format(item, os.path.join(path_destino, os.path.basename(item))))
