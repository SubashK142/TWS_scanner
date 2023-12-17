from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 4
num_epochs = 10

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    r'C:/Users/SUBASH/OneDrive/Desktop/tws/Backend/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs
)

@app.route('/get',methods=['GET'])
def g():
    print("works")
    return "subash"

@app.route('/image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'})

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = 'uploaded_image.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            size = os.path.getsize(filepath)
            print("Image size:", size, "bytes")
            from tensorflow.keras.preprocessing import image
            import numpy as np

            test_image = image.load_img(r'C:/Users/SUBASH/OneDrive/Desktop/tws/Backend/uploads/uploaded_image.jpg', target_size = (224,224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
            m=np.max(result)
            print(m)
            
            if(m<1):
                spe="copper head"
            else:
                spe="coral snake"
            return jsonify({
                'message': 'Image uploaded successfully',
                'filename': spe,
                'filepath': filepath
            })

        return jsonify({'error': 'Invalid file format'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0',port=5000,debug=True)









