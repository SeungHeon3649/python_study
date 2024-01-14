import pathlib
from keras.utils import image_dataset_from_directory

d_path = pathlib.Path("images")
train_data = image_dataset_from_directory(d_path, validation_split = 0.2, subset = "training",
                                    seed = 123, image_size = (224, 224), batch_size = 32)
val_data = image_dataset_from_directory(d_path, validation_split = 0.2, subset = "validation",
                                    seed = 123, image_size = (224, 224), batch_size = 32)
print(val_data)
