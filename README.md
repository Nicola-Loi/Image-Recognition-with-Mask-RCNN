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








