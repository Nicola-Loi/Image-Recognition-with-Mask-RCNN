# -*- coding: utf-8 -*-
"""Mask RCNN - runnato .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GvzkPgKbsJ-5kuY9a0wRckRl2lev_XG2

## **Mask R-CNN per Nike**

**Introduzione**

L’obiettivo della rete Mask R-CNN  è quello di rilevare e classificare gli oggetti in un immagine e simultaneamente generare una segmentation mask ad alta qualità per ogni elemento. La rete, detta Mask R-CNN, estende la Faster R-CNN aggiungendo un branch in parallelo con quelli gia esistenti. Il nuovo branch detto ‘Mask’ aggiunge la possibilità di  predirre delle  maschere di riconoscimento degli oggetti. I creatori della rete avevano come obiettivo quello di creare una rete che potesse migliorare le performance di reti come la Fast/Faster R-CNN o la Fully Convolutional Network (FCN) col fine di sviluppare un framework per la ‘Instance Segmentation’. L’ Instance Segmentation è tutt’ora una sfida aperta poiché essa richiede la corretta classificazione di tutti gli oggetti dell’immagine e allo stesso tempo la precisa segmentazione di essi. Essa infatti combina le classiche tasks per la object detection della computer vision, dove l’obiettivo è quello di classificare i singoli oggetti usando dei bounding box e unisce ad essa la Semantic Segmentation che ha come obiettivo classificare ogni pixel per un set di categorie fissate senza dipendere dalla rilevazione degli oggetti. Applicheremo questo framework ad un data set collezzionato e annotato mediante un software detto labelmg.
L' **Obiettivo** sarà quello di segmentare e riconoscere il logo Nike in immagini di prodotti e non.

 Ma prima andiamo a spiegare l'archittetura della rete.

![alt text](https://www.researchgate.net/profile/Lukasz_Bienias/publication/337795870/figure/fig2/AS:834563236429826@1575986789511/The-structure-of-the-Mask-R-CNN-architecture.png)

## Faster R-CNN

**Gli sviluppi**

Al fine di comprendere il funzionamento di Mask R-CNN, è opportuno evidenziare gli aspetti principali delle reti region-based CNN. Le reti region-based che precedono Mask R-CNN sono state proposte per risolvere problemi di object detection. La rete R-CNN Region-based CNN, proposta nel 2014,trasforma un problema di object detection in un problema di classificazione.Il funzionamento è il seguente: data un’immagine, vengono innanzitutto estratte circa 2000 possibili regioni d’interesse (RoI) mediante un algoritmo di Selective Search, successivamente verranno estratte le caratteristiche di ogni singola regione utilizzando una CNN ed infine, verranno classificate le regioni sulla base delle caratteristiche estratte. Il problema principale di R-CNN è dovuto al costo computazionale elevato. In seguito viene sviluppata Fast R-CNN che migliora i tempi di training e i risultati invertendo la scelta delle regioni d'interesse con la feature extraction. In seguito nasce Faster R-CNN la cui idea base  consiste nel migliorare le performance di Fast R-CNN modificando il processo di selezione delle regioni d’interesse.
Poichè le precedenti versioni utilizzavano un algoritmo di selective search esterno, e dunque non facendo parte della rete neurale vera e propria, esso non poteva essere allenato in fase di training. Faster R-CNN aggiunge ulteriori strati che permettono l’estrazione delle regioni d’interesse medianteuna rete convoluzionale vera e propria,Region Proposal Network(RPN), la quale sostituisce definitivamente l’algoritmo Selective Search. Analizziamo le singole parti della rete Mask-RCNN.

**Convolutional BackBone: ResNet**

La prima struttura a cui viene dato in pasto il nostro data set è una rete convoluzionale. Nell’articolo sono state utilizzate diverse architteture: ResNet50,ResNet-101,ResNetX. 
Nel nostro caso utilizzeremo la ResNet 101 con un estensione dei layer finali detta FPN (Feature Pyramid Network).
La ResNet101 è una rete convoluzionale con 101 layer che fa parte delle ResidualNets. Le ResNet sono delle reti che mediante uno stratagemma tentano di risolvere il problema del gradiente. Esso difattti diventa infinitamente piccolo aggiungendo molti layer. L’idea delle ResNet consiste quindi nell’introdurre un termine che viene detto “identity shortcut connection”.  Dove semplicemente viene sommata l’identità all’input di qualche layer successivo.  

![alt text](https://i.stack.imgur.com/msvse.png)

L’output di questa prima parte sarà una feature map. Spetta ora al RPN selezionare delle proposte di zone d’interesse e raffinare la selezione delle regioni, Tralasciamo la parte finale della rete per concentrarti sulla RPN.

**Regional Proposal Network (RPN)**

La rete Region Proposal Network(RPN) prende un’immagine (Feature Map) in input e restituisce in output un insieme di regioni proposte: a ciascuna di queste regioni è associata la probabilità che l’oggetto si trovi effettivamente in tale area. Con il termine ’regione’ la rete intende un’area avente dimensione rettangolare in cui è possibile rilevare un oggetto.Una finestra scorrevole avente dimensione $n×n$ viene fatta scorrere lungo la feature map al fine di determinare i riquadri delle region proposal, denotati anche come anchors. Ogni volta che viene fatta scorrere la finestra lungo la feature map, vengono individuate al più $k$ anchors al variare di differenti valori di scala e aspect ratio.Di default si hanno 3 differenti valori di scala e 3 differenti aspect ratio, per un totale di $k= 3·3 = 9$ anchors per ciascuna finestra scorrevole. Per ciascuna di queste regioni verrà associata la probabilità di contenere un’istanza (score e le coordinatee dell’anchor box individuato.Pertanto, per unafeature map avente dimensione $W×H$ verranno estratte $W·H·k$ anchor boxes.Una volta determinate le region proposal, esse verranno nuovamente elaborate dalla rete.
La loss function relativa ad ogni RoI estratta sarà uguale a: 

$J = J_{cls} + J_{box} $


![alt text](https://www.researchgate.net/profile/Jerome_Williams9/publication/322000654/figure/fig2/AS:651416800604161@1532321276378/Region-Proposal-Network-from-3-The-RPN-implements-a-sliding-window-over-a-CNNs.png)

**RoI Align**

Per capire la differenza tra il RoI Align e il RoI pooling facciamo un esempio. Supponiamo di avere un’immagine di dimensione $128×128$ ed una feature map ad essa associata avente dimensioni $25×25$. Vogliamo estrarre una RoI di dimensione $15×15$ dell’immagine originale.Pertanto, sarebbe necessario estrarre una regione di pixel dalla feature map pari a $m×m$, con $m=\frac{25∗15}{128}≈2.93 $. Con il RoI Pooling perderemo 0.93 pixel di informazione nella feature map mentre con RoI Align tramite una presa di punti di campionamente nella RoI e poi un interpolazione bilineare sarà possibile mantenere le dimensioni della RoI nella feature map senza perdere pixel d'informazione.

![alt text](https://cdn-images-1.medium.com/max/1000/1*OdUWLZq9M4iebhIjF-6Xkg.png)

A questo punto la feature map con le RoI evidenziate viene data in pasto al Mask Branch e la parte di Fully Connected Layers per la classificazione e la regressione.

**Mask Branch/ Classification / Bounding Box**

Riportiamo un immagine in cui viene riportara la cosidetta 'head' della rete in cui avviene la classificazione, la generazione dei bounding box (regressione) e delle maschere.

![alt text](https://d6vdma9166ldh.cloudfront.net/media/images/1525346771216-mask.jpg)

Si può notare come la differenza tra le due heads è il numero di layer di convoluzione per il mask branch e le varie dimensioni di filtri. Per la classificazione verrà usata una Softmax e per i bounding box un regressore. Mentre per il mask branch a ciascuna RoI viene associata una sola maschera ground truth e verrò applicata una funzione d’attivazione sigmoide ad ogni pixel della maschera. Il branch associato alla previsione della maschera genererà maschere binarie aventi dimensioni $m×m$ per ciascuna delle $K$ possibili classi. Pertanto, in totale si genereranno $K * m * m$ possibili maschere, ciascuna associata ad una diversa classe. In totale la loss function da minimizzare per ogni RoI sarà:

$ J=J_{cls} + J_{box} + J_{mask} $

dove le loss function per il classificatore e per il regressore del box saranno :

$J_{cls+box}( p_{i},t_{i}) = \frac{1}{N_{cls}}  \sum_{i}  J_{cls} (p_{i},p^{*}_{i})  +\frac{\lambda}{N_{box}} \sum_{i} p^{*}_{i} *smooth_{L_{1}}  (t_{i}-t^{*}_{i})$

dove $J_{cls}$ sarà la cross-entropy calcolata su due classi (si può facilmente passare da unproblema di classificazione multiclasse ad un problema di classificazione binario considerando la probabilità che l’i-esima ancora appartenga ad una determinata classe contro la probabilità che non vi appartenga).

- $p_{i}$:  probabilità predetta che l’i-esima ancora appartenga ad un oggetto;

- $p^{*}_{i}$:  probabilità ground truth che l’i-esima ancora sia un oggetto;

- $t_{i}$: coordinate predette;

- $t^{*}_{i}$:  coordinate ground truth;

- $N_{cls}, N_{box}$ termini di normalizzazione relativi, rispettivamente, a $J_{cls}$ e $J_{box}$;

- $\lambda$ parametro che bilancia $J_{cls}$ e $J_{box}$;

- la funzione $smooth_{L_{1}}$ applicherà un termine di smooth alle coordinate del bounding box a seconda che la differenza tra la coordinata predetta e quella ground truth sia maggiore o minore di 1.

Relativamente al calcolo di $L_{mask}$ , a ciascuna RoI viene associata una sola maschera ground truth e verrà applicata una funzione d’attivazione sigmoide ad ogni pixel della maschera. Il branch associato alla previsione della maschera genererà maschere binarie aventi dimensioni $m×m$ per ciascuna delle $K$ possibili classi. Pertanto, in totale si genereranno $K·m^{2}$ possibili maschere, ciascuna associata ad una diversa classe.$L_{mask}$ viene definita come la media tra le binary cross-entropy loss function, in cui è inclusa la $k$-esima maschera se la regione è associata alla $k$-esima maschera ground truth:

$J_{mask}=-\frac{1}{m^{2}}\sum_{1<i,j<m}[y_{ij}log(y^{*k}_{ij}) +(1-y_{ij})log(1-y^{*k}_{ij}) ]$
"""

#Clono la repository di Mask_RCNN
!git clone https://github.com/matterport/Mask_RCNN

"""What about changing: self.keras_model.metrics_tensors.append(loss) to: self.keras_model.add_metric(loss, name)"""

# Commented out IPython magic to ensure Python compatibility.
#Importo tensorflow e setto le versioni da usare
# %tensorflow_version 1.x
import tensorflow as tf
import keras
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# # Installo i setup
# cd Mask_RCNN
# python setup.py install

# Commented out IPython magic to ensure Python compatibility.
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline 

# Directory del progetto
ROOT_DIR = os.path.abspath("./Mask_RCNN/")

# Importo la directory sul sistema
sys.path.append(ROOT_DIR)  # To find local version of the library

#Importi alcuni moduli che mi serviranno per far girare il modello
#e per visualizzare i risultati
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize


#Importo la configurazione COCO, mi serviranno per utilizzare i pesi
#che andrò a caricare
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))

  # find local version
import coco

# Directory su cui salvare la rete
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path destino ai pesi 
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Faccio il Download dei pesi e gli indirizzo al path dei pesi
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

"""## DataSet e MSCOCO

Nella parte di codice precedente sono stati importati i pesi per sfruttare il transfer learning. La rete infatti essendo molto grande impegherebbe molto tempo per essere riallenata su un nuovo data set. I creatori hanno allenato la rete su un dataset di immagini chiamato MSCOCO.

**MSCOCO**

MSCOCO è un dataset composto da più di 330K immagini di persone, animali, strade etc. Il data set è annotato con stile Pascal VOC e Yolo ed è stato costruito con obiettivi di Object Detection e Recognition in context. Le immagini presentano dei Bounding Box per gli elementi presenti e delle maschere. Sono presenti più di 91 categorie. 

**NikeDataSet**

Il dataset che utilizzeremo per il progetto consiste in 150 immagini di prodotti e non che mostrano il logo Nike. Il fine infatti sarà quello di segmentare e riconoscere il logo nelle diverse immagini. Il dataset è stato creato andando a scaricare semplicemente le immagini da google e creando un folder. In seguito le immagini sono state annotate con un software chiamato **labellmg** che annota le dimensioni e le posizioni del bounding box con stile PascalVOC.

Il nostro dataset essendo molto piccolo verrà splittato in 100 immagini per il training, 20 per il validation e 30 per il test set.
"""

!git clone https://github.com/NikoLopez/nikedata.git

from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset

#Definisco alcune funzioni che mi permettono di caricare e generare i dataset per il training/validation/test.
#Di estrarre le dimensione dei box e le maschere.

class NikeDataset(Dataset):

    def load_dataset(self, dataset_dir, name):
        # define one class
        self.add_class("dataset", 1, "nike")
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annotations/'
        # find all images
        train_list = [*range(0, 100, 1)] 
        val_list=[*range(100,120,1)]
        test_list=[*range(121,150,1)]
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            if name =='train' and int(image_id) not in train_list :
                continue
            if name =='val' and int(image_id) not in val_list : 
                continue 
            if name =='test' and int(image_id) not in test_list : 
                continue 
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height
 
    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('nike'))
        return masks, asarray(class_ids, dtype='int32')
 
    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id] 
        return info['path']

# train set
train_set = NikeDataset()
train_set.load_dataset('nikedata', 'train' )
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
#val set

val_set =NikeDataset()
val_set.load_dataset('nikedata', 'val')
val_set.prepare()
print('Val: %d' % len(val_set.image_ids))

# test set
test_set =NikeDataset()
test_set.load_dataset('nikedata', 'test')
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

import matplotlib.pyplot as pyplot
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances

#Esempio di un immagine del dataset con bounding box e maschera annessa

# load an image
image_id = 11
image = train_set.load_image(image_id)
print(image.shape)
# load image mask
mask, class_ids = train_set.load_mask(image_id)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names)

"""# Training & Parameter Selection

Per allenare la rete a classificare e segmentare il logo nike nelle immagini che gli daremo in pasto proveremo diverse configurazioni. In tutte le prove che seguiranno sono stati allenati solo i layer 'head'. I parametri che andremo a cambiare saranno il numero di epoche, gli step per epoca, il learning rate e il parametro di weight decay. Il Learning Momentum non verrà cambiato perchè risulta essere già al massimo da setup. Per visualizzare l'andamento della LossFunction e quindi il nostro errore sul TrainSet e ValSet utilizzeremo Tensorboard. 

Si noti come la configurazione presenti molti più parametri che potrebbero essere cambiati, come ad esempio il ridimensionamento dell'immagine, la numero degli anchors e degli aspect ratio nel RPN etc. Tuttavia queste modifiche comportano un riallenamento totale della rete che risulta computazionalmente impensabile.

Premettiamo che Tensorboard non permette di visualizzare i titoli degli assi. Nell'asse x abbiamo il numero di epoche e nell'asse y il valore della LossFunction.

Modello 0: Learning Rate 0.001 / Epoche: 20  / Step = #di immagini / Weight Decay:0.0001
"""

from mrcnn.config import Config

# define a configuration for the model
class NikeConfig(Config):
  NAME='nike_cfg'
  # Number of classes (background + logo)
  NUM_CLASSES = 1 + 1
  IMAGES_PER_GPU = 1 

  STEPS_PER_EPOCH = 99
  VALIDATION_STEPS = 20

  BACKBONE = "resnet101"
  BACKBONE_STRIDES = [4, 8, 16, 32, 64]
  FPN_CLASSIF_FC_LAYERS_SIZE = 1024
  TOP_DOWN_PYRAMID_SIZE = 256

  RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
  RPN_ANCHOR_RATIOS = [0.5, 1, 2]
  RPN_ANCHOR_STRIDE = 1
  RPN_NMS_THRESHOLD = 0.7
  RPN_TRAIN_ANCHORS_PER_IMAGE = 256

  IMAGE_RESIZE_MODE = "square"
  IMAGE_MIN_DIM = 800
  IMAGE_MAX_DIM = 1024
  IMAGE_CHANNEL_COUNT = 3

  LEARNING_RATE = 0.001
  LEARNING_MOMENTUM = 0.9

  WEIGHT_DECAY = 0.0001

# prepare config
config = NikeConfig()
model = modellib.MaskRCNN(mode='training', model_dir=MODEL_DIR, config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model.train(train_set, val_set, learning_rate=config.LEARNING_RATE,  epochs=20, layers='heads')

"""Andiamo a visualizzare i plot degli andamenti di TestLossFunction e ValLossFunction in funzione delle epoche del train. I plot che verranno visualizzati sono quelli della Loss totale:


$ J=J_{cls} + J_{box} + J_{mask} $
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir=/content/Mask_RCNN/logs/nike_cfg20200615T0924

from IPython.display import Image
Image(filename='model0_loss.jpg')

from IPython.display import Image
Image(filename='model0_val_loss.jpg')

"""Modello 1: Learning Rate 0.01 Epoche 20 Step = #di immagini Weight Decay:0.0001"""

from mrcnn.config import Config

# define a configuration for the model
class NikeConfig1(Config):
  NAME='nike_cfg1'
  # Number of classes (background + logo)
  NUM_CLASSES = 1 + 1
  IMAGES_PER_GPU = 1 

  STEPS_PER_EPOCH = 99
  VALIDATION_STEPS = 20

  LEARNING_RATE = 0.01
  LEARNING_MOMENTUM = 0.9

  WEIGHT_DECAY = 0.0001

# prepare config
config1 = NikeConfig1()
model1 = modellib.MaskRCNN(mode='training', model_dir=MODEL_DIR, config=config1)
# load weights (mscoco) and exclude the output layers
model1.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model1.train(train_set, val_set, learning_rate=config1.LEARNING_RATE,  epochs=20, layers='heads')

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir=/content/Mask_RCNN/logs/nike_cfg120200615T0939

from IPython.display import Image
Image(filename='model1_loss.jpg')

from IPython.display import Image
Image(filename='model1_val_loss.jpg')

"""Modello 2: Learning Rate 0.001 Epoche 15 Step = 2* #di immagini Weight Decay:0.0001"""

from mrcnn.config import Config

# define a configuration for the model
class NikeConfig2(Config):
  NAME='nike_cfg2'
  # Number of classes (background + logo)
  NUM_CLASSES = 1 + 1
  IMAGES_PER_GPU = 1 

  STEPS_PER_EPOCH = 200
  VALIDATION_STEPS = 40

  LEARNING_RATE = 0.001
  LEARNING_MOMENTUM = 0.9

  WEIGHT_DECAY = 0.0001

# prepare config
config2 = NikeConfig2()
model2 = modellib.MaskRCNN(mode='training', model_dir=MODEL_DIR, config=config2)
# load weights (mscoco) and exclude the output layers
model2.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model2.train(train_set, val_set, learning_rate=config2.LEARNING_RATE,  epochs=15, layers='heads')

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir=/content/Mask_RCNN/logs/nike_cfg220200615T1018

from IPython.display import Image
Image(filename='model2_loss.jpg')

Image(filename='model2_val_loss.jpg')

"""Modello 3: Learning Rate 0.001 Epoche 10 Step = #di immagini Weight Decay:0.001"""

from mrcnn.config import Config

# define a configuration for the model
class NikeConfig3(Config):
  NAME='nike_cfg3'
  # Number of classes (background + logo)
  NUM_CLASSES = 1 + 1
  IMAGES_PER_GPU = 1 

  STEPS_PER_EPOCH = 99
  VALIDATION_STEPS = 20

  LEARNING_RATE = 0.001
  LEARNING_MOMENTUM = 0.9

  WEIGHT_DECAY = 0.001

# prepare config
config3 = NikeConfig3()
model3 = modellib.MaskRCNN(mode='training', model_dir=MODEL_DIR, config=config3)
# load weights (mscoco) and exclude the output layers
model3.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model3.train(train_set, val_set, learning_rate=config3.LEARNING_RATE,  epochs=10, layers='heads')

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir=/content/Mask_RCNN/logs/nike_cfg320200615T1049

Image(filename='model3_loss.jpg')

Image(filename='model3_val_loss.jpg')

"""Modello 4: Learning Rate 0.001 Epoche 15 Step = #di immagini Weight Decay:0.001"""

from mrcnn.config import Config

# define a configuration for the model
class NikeConfig4(Config):
  NAME='nike_cfg4'
  # Number of classes (background + logo)
  NUM_CLASSES = 1 + 1
  IMAGES_PER_GPU = 1 

  STEPS_PER_EPOCH = 99
  VALIDATION_STEPS = 20

  LEARNING_RATE = 0.001
  LEARNING_MOMENTUM = 0.9

  WEIGHT_DECAY = 0.001

# prepare config
config4 = NikeConfig4()
model4 = modellib.MaskRCNN(mode='training', model_dir=MODEL_DIR, config=config4)
# load weights (mscoco) and exclude the output layers
model4.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model4.train(train_set, val_set, learning_rate=config4.LEARNING_RATE,  epochs=15, layers='heads')

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir=/content/Mask_RCNN/logs/nike_cfg420200615T1116

Image(filename='model4_loss.jpg')

Image(filename='model4_val_loss.jpg')

"""Dall'analisi degli andamenti della Loss Function per il training e per il validation deduciamo che il modello 3 sia il migliore per approssimazione e generalizzazione. Esso infatti a differenza degli altri modelli presenta una validation loss che decresce con l'aumentare delle epoche, sintomo di capacità di generalizzazione. Gli altri infatti presentano chiari sintomi di overfitting. Bisogna inoltre notare che le tutte le loss fluttuano parecchio. Questo potrebbe essere dovuto al fatto che la dimensione del data set è piccola.

# Model Evaluation

Per la validazione del modello andiamo a spiegare le metriche che vengono utilizzate del campo della Instance segmentation:

**IoU(Intersection over union)**

E' il rapporto tra l'area di intersezione (calcolata andando a valutare il numero di pixel comuni tra la maschera predetta e quella ground truth)  e l'area di unione tra il ground truth box( box target) e il box predetto. Per capire meglio riportiamo un immagine in cui il box rosso rappresenta quello predetto e quello verde quello ground truth.


![alt text](https://miro.medium.com/max/1400/1*7ub564dUk-dupk1JLCp-Jg.jpeg)


**Precision-Recall**

Oltre a determinare bounding box e maschera di ogni singola istanza,Mask R-CNN deve saper classificare correttamente l’istanza individuata.Più in generale, in un problema di classificazione, è necessario stabilire se un input x avente output y venga classificato correttamente dal modello. A seconda della classe che verrà restituita in output, si potrà determinare se il dato in input sia stato classificato correttamente. E' necessario distinguere i seguenti casi possibili:

•True Positive (TP): è un esempio che viene classificato correttamente dal modello come appartenente ad una classe;

•True Negative (TN): è un esempio che viene classificato correttamente dal modello come non appartenente ad una classe;

•False Positive (FP): è un esempio che viene classsificato in maniera non corretta dal modello come appartenente ad una classe;

•False Negative (FN): è un esempio che viene classificato in maniera incorretta dal modello come non appartenente ad una classe.

Da questa casistica di classificazione possiamo calcolare due quantità fondamentali che ci permetteranno di stimare la bonta della nostra rete

![alt text](https://miro.medium.com/max/888/1*7J08ekAwupLBegeUI8muHA.png)

**Link IoU-PR**

Nel problemi di Instace Segmentation un istanza verrà classificata True Positive, False Positive, o False Negative (bisogna notare che non esistono TrueNegative perchè viene assunto che il bounding box avrà sempre qualcosa da contenere e quindi non sarà mai vuoto) a seconda del valore di IoU. Tipiacamente viene definita una soglia pari a un IoU=0.5. Per cui:

- se IoU > 0.5 è un TruePositive,

- se IoU < 0.5 è un FalsePositive,

- se IoU > 0.5 ma l'oggetto è misclassificato sarà un FalseNegative.

**Average-Precision**

Per rappresentare opportunamente la capacità di classificazione del modello,le due metriche vengono valutate nella cosiddetta curva precision-recall, la quale avrà nelle ascisse i valori relativi alla recall e nelle ordinate i valori relativi alla precision. L’area sottesa dalla curva precision-recall è detta average precision(AP) ed assumerà un valore compreso tra 0 ed 1. Essa sarà dunque uguale a:

$AP=\int^{1}_{0} p(r) dr $

![alt text](https://miro.medium.com/max/4000/1*naz02wO-XMywlwAdFzF-GA.jpeg)

Riporteremo quindi per ogni immagine la curva Precision-Recall e il valore dell'AveragePrecision. E poi per stimare la performance media del modello ne faremo una media sui valori per ogni immagine.


**Confusion Matrix & Match**

Un altro metodo di visualizzazione delle performance del modello nei problemi di Instance Segmentation è l'utilizzo di una Confusion Matrix che mi riporta il match o non match dell'istanza. La matrice è costruita con un numero di colonne pari al numero di istanze ground truth ( numero di istanze che veramente ci sono) e come numero di righe il numero di istanze predette. Ogni elemento della matrice riporterà il valore di IoU e quindi l'effettivo match. Inoltre affianco agli elementi di riga verrà riportata la probabilità con cui è stata predetta la classe.
"""

import mrcnn.model as modellib
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean

## Creo il modello per la valutazione delle performance e pongo il set per il 'Test' quindi lo imposto in 'Inference'
model_test = modellib.MaskRCNN( mode="inference", model_dir=MODEL_DIR, config = config3)

# Carico i pesi generati dal training del modello3 sul mio data set
model_test.load_weights('/content/Mask_RCNN/logs/nike_cfg320200615T1049/mask_rcnn_nike_cfg3_0010.h5', by_name=True)

"""Definisco due funzioni che mi permetteranno di valutare le performance del mio modello.
- *show_results*: mostra il risultato di 'instance segmentation' su un immagine presa da un data set. Mi riporta il grafico di Precision-Recall con il relativo valore di AP con soglia di IoU pari a 0.5 per l'immagine e mi plotta la confusion Matrix.

- *Mean_AP*: mi calcola la media su tutto il data set scelto delle AP delle singole immagini.
"""

def show_results(image_id,dataset,config,model):
  image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, config, image_id, use_mini_mask=False)
  results = model.detect([image], verbose=1)
  r = results[0]  
  print('                                                        ')
  visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], 
                            title="Predictions",  figsize=(10,10)),
  AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'])
  print('                                                        ')
  visualize.plot_precision_recall(AP, precisions, recalls)
  print('                                                        ')
  visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],overlaps, dataset.class_names)

def Mean_AP(dataset,config,model):
  APs=[]
  for image_id in dataset.image_ids:

    if image_id==18:
      continue
    if image_id==21:
      continue
    
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, config, image_id, use_mini_mask=False)
    results = model.detect([image], verbose=0)
    r = results[0]  
    AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
  return mean(APs)

"""Calcolo delle performance per un immagine del TrainSet"""

show_results(13,train_set,config3,model_test)

"""Calcolo delle performance per un immagine del TestSet"""

show_results(2,test_set,config3,model_test)

print('Media su tutti i AP delle immagini del TestSet',Mean_AP(test_set,config3,model_test))
print('Media su tutti i AP delle immagini del TrainSet',Mean_AP(train_set,config3,model_test))

"""# Conclusioni

Per concludere questo breve progetto sull'utilizzo di Mask RCNN per la segmentazione di oggetti in un immagine possiamo affermare che la rete riesce a rilevare e segmentare il logo della nike nelle immagini del TestSet con una precisione media di circa 0.43 sul Test Set e una precisione del 0.89 sul TrainSet. Questo risultato è chiaramente legato al fatto che i bounding box generati risultano essere giusti ma la segmentazione delle immagine talvolta non è precisa e qualche volta la rete rileva sezioni dell'immagine che non racchiudono il logo Nike. Il risultato non raggiunge assolutamente la precisione di circa 0.6 sul Test Set ottenuta dai creatori della rete nella challenge di MSCoco.  Tuttavia essendo il dataset molto piccolo e le possibilità computazionali  ridotte il risultato risulta essere abbastanza buono. Possiamo notare però che nel TrainSet raggiungiamo una precisione abbastanza alta. 

In vista di un implementazione futura si potrebbe ingrandire il dataset scaricando e annotando nuove immagini o facendo data augmentation. Inoltre si potrebbe utilizzare un'altra archittetura di rete come la 'resnet50' o la 'resnetX'. Oppure si potrebbero ridurre le dimensioni delle immagini diminuendo il padding. Oppure si potrebbe riallenare tutta la rete e non solo i layer finali.


Bibliografia:
- https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/inspect_model.ipynb

- Mask R-CNN, Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick,  	arXiv:1703.06870

- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, arXiv:1506.01497 

- https://towardsdatascience.com/region-proposal-network-a-detailed-view-1305c7875853

- https://towardsdatascience.com/understanding-region-of-interest-part-2-roi-align-and-roi-warp-f795196fc193

- https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
"""
