import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image parameters
img_size = 224
batch_size = 16

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_data = train_datagen.flow_from_directory(
    "chest_xray/train",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary"
)

# Load validation data
val_data = val_datagen.flow_from_directory(
    "chest_xray/val",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary"
)

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = True

# Freeze only early layers
for layer in base_model.layers[:100]:
    layer.trainable = False
    
# Build model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
# Train model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Save model
model.save("xray_model.h5")

print("Model training complete. Saved as xray_model.h5")
