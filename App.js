
import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, Button, Image,StatusBar } from 'react-native';
import { Camera } from 'expo-camera';
import axios from 'axios';


  export default function App() {
    const [hasCameraPermission, setHasCameraPermission] = useState(null);
    const [camera, setCamera] = useState(null);
    const [image, setImage] = useState(null);
    const [type, setType] = useState(Camera.Constants.Type.back);
    const [data, setData] = useState(null);
    
    const [error, setError] = useState(null);

    const [loading, setLoading] = useState(false);

    // ... (other functions)
    const postImage = async (photo) => {
      const formData = new FormData();
      formData.append('image', {
        uri: photo,
        type: 'image/jpeg',
        name: 'image.jpg',
      });
    
      try {
        const response = await fetch('http://192.168.86.197:5000/image', {
          method: 'POST',
          body: formData,
        });
    
        const data = await response.json();
        
        console.log('Response from server:', data.filename);
      } catch (error) {
        console.error('Error uploading image:', error);
      }
    };

    useEffect(() => {
      (async () => {
        const cameraStatus = await Camera.requestCameraPermissionsAsync();
        setHasCameraPermission(cameraStatus.status === 'granted');
      })();
    }, []);

    const takePicture = async () => {
      if (camera) {
        const data = await camera.takePictureAsync(null);
        setImage(data.uri);
        console.log(data.uri);
        postImage(data.uri)
      }
    };
    
    const switchCameraType = () => {
      setType(
        type === Camera.Constants.Type.back
          ? Camera.Constants.Type.front
          : Camera.Constants.Type.back
      );
    };

    const retakePicture = () => {
      setImage(null);
    };

    if (hasCameraPermission === false) {
      return <Text>No access to camera</Text>;
    }

    return (
      <View style={styles.container}>
      <StatusBar backgroundColor="#3498db" barStyle="light-content" />
      <Text style={styles.text}>TWS SCANNER</Text>
     
        {!image ? (
          <View style={styles.cameraContainer}>
            <Camera
              ref={(ref) => setCamera(ref)}
              style={styles.fixedRatio}
              type={type}
              ratio={'1:1'}
            />
          </View>
        ) : (
          <View style={styles.imageContainer}>
            <Image source={{ uri: image }} style={styles.image} />
            
            <View style={styles.buttonContainer1}>
            <Button title="Retake Picture" onPress={retakePicture} />
            <Button title="GET DATA" onPress={takePicture} />

            <View>
        
          </View>
          </View>
          </View>
          
        )}

        {!image && (
          <View style={styles.buttonContainer}>
            <Button title="Flip Camera" onPress={switchCameraType} />
            <Button title="Take Picture" onPress={takePicture} />
          </View>
        )}
      </View>
    );
  }

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  cameraContainer: {
    flex: 1,
    flexDirection: 'row',
    marginTop: 70,
    padding:10,
    
    
  },
  fixedRatio: {
    flex: 1,
    aspectRatio: 1,
    
  },
  imageContainer: {
    flex: 1,
    paddingTop:'-10',
    padding:10,
    width:400,
  },
  image: {
    flex: 1,
    width: '100%',
    resizeMode:"center"
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 10,
    padding:80,
    
  },
  buttonContainer1: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingTop:30,
    padding:60,
  },
  text:{
    fontSize:35,
    fontWeight:'bold',
    textAlign:'center',
    padding:20
  },
  button:{
    width: 100,  // Set the desired width
    height: 50
  }
});