# Sinais
## Projeto de Sinais e Sistemas (ES413) - Detecção de rótulos de risco

### Alunos: 
  Jeferson Severino de Araujo (jsa2)
  
  Lucas Nascimento Brandão (lnb)
  
  Marcus Vinicius de Faria Santos (mvfs)
  
  Matheus Julio Boncsidai de Oliveira (mjbo)
  
  Rodrigo Rocha Moura (rrm2)

## Como rodar?
Para esse projeto, foi utilizado como base o repositório Jetson Inference. Nele, existe uma suite de aplicações de Inteligência Artificial voltadas para a os computadores embarcados Jetson. No caso, a rede que utilizamos é a SSD MobileNetV2, para object detection.

Link da jetson: <https://drive.google.com/file/d/1lka6pKbZe9D9FSJJBoi7xkpgdfeKczwd/view?usp=share_link>

Os modelos treinados estão em "Sinais/jetson/python/training/detection/ssd/models/hm13/", sendo a v1 a versão sem augmentation, a v2 com os filtros e transformadas que escolhemos e a v3 augmentation do repositório original do dataset. Caso todas as bibliotecas para rodar no desktop estejam instaladas, basta entrar em uma das pastas (v1, v2, v3) e rodar o arquivo main.py. Ele vai pegar um vídeo e gerar outro como resposta.

## Filtros e transforamda
O código dos filtros e das transformadas se encontram nas pastas:

Sinais/Est_Filter

Sinais/FFT_Filter 

## Como treinar?
Vá até a pasta "Sinais/jetson/python/training/detection/ssd", execute os seguintes comandos para garantir que tenha as Libs necessárias:

pip install -r requirements.txt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117 

pip install opencv-python

Após isso, basta executar:

python3 train_ssd.py --dataset-type=voc --data=data/drones --model-dir=models/drones --epochs 120 --resume="path/to/checkpoint.pth"

## Dataset
O dataset está na pasta:

Sinais/jetson/python/training/detection/ssd/data/hm13

## Dúvidas
Dependendo  de onde será rodado, o código pode demandar algumas libs diferentes. Em caso de dúvidas, entre em contato:
mvfs@cin.ufpe.br
