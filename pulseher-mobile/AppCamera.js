// AppCamera.js
import React, { useState, useEffect, useRef } from "react";
import { View, Text, Button, Image, StyleSheet, Alert } from "react-native";
import { Camera, CameraType } from "expo-camera";
import Constants from 'expo-constants';

// Determine API base via multiple sources for flexibility in Expo
// Priority: app config extra -> environment variable -> global override -> fallback placeholder
const getApiBase = () => {
  // 1) Expo config extra (app.json/app.config.js) -> expoConfig.extra.EXPO_PUBLIC_API_BASE
  const expoExtra = (Constants.expoConfig && Constants.expoConfig.extra) || (Constants.manifest && Constants.manifest.extra);
  if (expoExtra && expoExtra.EXPO_PUBLIC_API_BASE) return expoExtra.EXPO_PUBLIC_API_BASE;

  // 2) Process env (when available in some setups)
  try {
    if (global && global.process && global.process.env && global.process.env.EXPO_PUBLIC_API_BASE) {
      return global.process.env.EXPO_PUBLIC_API_BASE;
    }
  } catch (e) {}

  // 3) Developer override (helpful for quick edits)
  if (global && global.__API_BASE) return global.__API_BASE;

  // 4) Fallback - user must replace with their PC LAN IP
  return "http://192.168.X.X:5000/api/ppg/upload-frame";
};

const API_BASE = getApiBase();

export default function AppCamera({ onMetrics }) {
  const [hasPermission, setHasPermission] = useState(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [lastFrame, setLastFrame] = useState(null);
  const cameraRef = useRef(null);
  const [torchOn, setTorchOn] = useState(true);
  const intervalRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === "granted");
    })();
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  useEffect(() => {
    let intervalId;
    if (isCapturing) {
      intervalId = setInterval(async () => {
        if (cameraRef.current) {
          try {
            const photo = await cameraRef.current.takePictureAsync({
              base64: true,
              quality: 0.5,
              skipProcessing: true,
            });
            setLastFrame(photo.uri || null);

            // POST base64 to backend
            await fetch(API_BASE, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ frame_data: photo.base64 })
            });
          } catch (err) {
            console.error("Capture error:", err);
          }
        }
      }, 200); // capture every 200 ms (~5 FPS)
    }
    return () => clearInterval(intervalId);
  }, [isCapturing]);

  if (hasPermission === null) return <View style={styles.info}><Text>Requesting camera permission...</Text></View>;
  if (hasPermission === false) return <View style={styles.info}><Text>No access to camera</Text></View>;

  return (
    <View style={styles.container}>
      <Camera
        style={styles.camera}
        type={CameraType.back}
        ref={cameraRef}
        ratio="16:9"
        flashMode={torchOn ? Camera.Constants.FlashMode.torch : Camera.Constants.FlashMode.off}
      />
      <View style={styles.controls}>
        <Button
          title={isCapturing ? "Stop" : "Start"}
          onPress={() => setIsCapturing(!isCapturing)}
        />
        <Button
          title={torchOn ? "Torch On" : "Torch Off"}
          onPress={() => setTorchOn(!torchOn)}
        />
        {lastFrame && (
          <Image source={{ uri: lastFrame }} style={styles.preview} />
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  camera: { flex: 4 },
  controls: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#111",
  },
  preview: { width: 100, height: 60, marginTop: 10 },
  info: { flex: 1, justifyContent: 'center', alignItems: 'center' }
});
