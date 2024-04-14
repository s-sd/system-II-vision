import tensorflow.keras as keras
from tensorflow.keras import layers

def get_alexnet_shared(img_size, label_size, num_classes):
    input_img = keras.Input(shape=img_size)
    input_label = keras.Input(shape=label_size)
    
    conv_1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')
    pool_1 = layers.MaxPooling2D((2, 2))
    conv_2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')
    pool_2 = layers.MaxPooling2D((2, 2))
    conv_3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')
    pool_3 = layers.MaxPooling2D((2, 2))
    
    x = conv_1(input_img)
    x = pool_1(x)
    x = conv_2(x)
    x = pool_2(x)
    x = conv_3(x)
    x = pool_3(x)
    
    y = conv_1(input_label)
    y = pool_1(y)
    y = conv_2(y)
    y = pool_2(y)
    y = conv_3(y)
    y = pool_3(y)
    
    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    y = layers.Flatten()(y)
    
    z = layers.concatenate([x, y])
    
    z = layers.Dense(512, activation='relu')(z)
    
    # Output layer
    outputs = layers.Dense(num_classes)(z)
    
    # Create the model
    model = keras.Model(inputs=[input_img, input_label], outputs=outputs)
    return model

def get_alexnet(img_size, label_size):
    # Image branch
    input_img = keras.Input(shape=img_size)
    x_img = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)
    x_img = layers.MaxPooling2D((2, 2))(x_img)
    x_img = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x_img)
    x_img = layers.MaxPooling2D((2, 2))(x_img)
    x_img = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x_img)
    x_img = layers.MaxPooling2D((2, 2))(x_img)
    x_img = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x_img)
    x_img = layers.MaxPooling2D((2, 2))(x_img)
    x_img = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x_img)
    x_img = layers.MaxPooling2D((2, 2))(x_img)
    x_img = layers.Flatten()(x_img)

    # Label branch
    input_label = keras.Input(shape=label_size)
    y_label = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_label)
    y_label = layers.MaxPooling2D((2, 2))(y_label)
    y_label = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(y_label)
    y_label = layers.MaxPooling2D((2, 2))(y_label)
    y_label = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(y_label)
    y_label = layers.MaxPooling2D((2, 2))(y_label)
    y_label = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(y_label)
    y_label = layers.MaxPooling2D((2, 2))(y_label)
    y_label = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(y_label)
    y_label = layers.MaxPooling2D((2, 2))(y_label)
    y_label = layers.Flatten()(y_label)

    # Concatenate features from both branches
    combined_features = layers.concatenate([x_img, y_label])

    # Fully connected layers
    z = layers.Dense(512, activation='relu')(combined_features)
    z = layers.Dense(256, activation='relu')(z)
    z = layers.Dense(64, activation='relu')(z)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(z)

    # Create the model
    model = keras.Model(inputs=[input_img, input_label], outputs=outputs)
    return model