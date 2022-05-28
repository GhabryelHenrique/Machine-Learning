import tensorflow as tf
import os, glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Diretório com as imagens de cavalos e humanos para treinamento 
train_house_dir = os.path.join('C:\\horses\\')
train_humans_dir = os.path.join('C:\\humans\\')

model = tf.keras.models.Sequential([
    # Formaro do input é de 300x300 com 3 bytes de cor cor

    # Primeira convolução
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Segunda convolução
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Terceira convolução
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Quarta convolução
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Quinta convolução
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Achatar os resultado em uma DNN (Deep Neural Network -  processar dados e reconhecer objetos visualmente)
    tf.keras.layers.Flatten(),

    # 512 neurons de camada oculta
    tf.keras.layers.Dense(512, activation='relu'),

    # 1 neuron de output. Tem um valor de 0-1 onde 0 é para cavalado e 1 para humano (já que temos somente 2 classes)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Faz um print da nossa rede neural que se encontra em model_plot.png
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Configurar as especificações para o treinamento do modelo
model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])

# Mudar a escala de todas as imagens para 1/255
train_datagen = ImageDataGenerator(rescale=1/255)

# Imagens de treinamento de fluxo em lotes de 128 usando o train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'C:\\Users\\ghabr\\Área de Trabalho\\machine learning\\Machine-Learning\\Dataset\\',  # Diretório de imagens de treinamento
        target_size=(300, 300),  # Todas as imagens em 300x300
        batch_size=128,
        # estamos usando o binary_crossentropy loss, então precisamos de rótulos binários
        class_mode='binary')

# Fazendo o Treinamento
history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=2)

model.save('horses_humans_model.h5')

# Mostrar como a precisão evolui ao longo dos períodos de treinamento
plt.plot(history.history['accuracy'], c='c', lw=2)
plt.grid(True)
plt.title('Precisão de Treinamentos com epochs\n', fontsize=15)
plt.xlabel('Treinamento epochs', fontsize=15)
plt.ylabel('Accuracy de treinamento', fontsize=15)

# Fazendo uma Predição com nosso modelo
path_img = 'C:\\Users\\ghabr\\Área de Trabalho\\machine learning\\Machine-Learning\\Teste\\'

# Para pegar todas as imagens da pasta
for arquivo in glob.glob(os.path.join(path_img, '*')):

    # Loanding da Imagem
    img = tf.keras.utils.load_img(arquivo, target_size=(300, 300))

    # Mostrar na tela qual a imagem que está sendo analizada
    plt.imshow(img)
    plt.show()
    
    # Arrumando a imagem para a rede ler
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    
    # Fazendo uma Predição com nosso modelo
    classes = model.predict(images, batch_size=10)
    
    # Se for humano ele mostra que encontrou humano e se for cavalo ele mostra q encontrou cavalo
    if classes[0]>0.5:
        print(arquivo.split('\\')[-1] + "Humano")
    else:
        print(arquivo.split('\\')[-1] + "Cavalo")
