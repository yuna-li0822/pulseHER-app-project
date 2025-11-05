PulseHER Vision Camera Example (bare React Native)

This folder contains scaffold and instructions to build a bare React Native project using
react-native-vision-camera and a native frame processor to compute per-frame green-channel
values and post numeric PPG values to your backend.

Important: vision-camera requires native builds. Do NOT run inside Expo Go. Use Xcode to
build on a device or emulator (iOS) and Android Studio for Android.

Quick steps (high level):

1) Create a new React Native app (bare) using React Native CLI:
   npx react-native init pulseher-vision
   cd pulseher-vision

2) Install required packages:
   yarn add react-native-vision-camera react-native-reanimated
   yarn add @shopify/react-native-skia # optional for drawing

3) iOS: install pods
   cd ios && pod install && cd ..

4) Follow vision-camera docs to implement a native frame-processor plugin.
   See: https://mrousavy.com/react-native-vision-camera/docs

5) Example usage in JS (in your app):

// App.js (snippet)
import React, {useEffect, useState} from 'react';
import {SafeAreaView, Text} from 'react-native';
import {Camera, useCameraDevices, useFrameProcessor} from 'react-native-vision-camera';
import {runOnJS} from 'react-native-reanimated';

export default function App() {
  const devices = useCameraDevices();
  const device = devices.back;
  const [ppg, setPpg] = useState(null);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet'
    // call native frame processor that returns normalized mean green
    const meanG = __detectMeanGreen(frame); // native function from plugin
    runOnJS(setPpg)(meanG);
  }, []);

  if (!device) return null;
  return (
    <SafeAreaView style={{flex:1}}>
      <Camera style={{flex:1}} device={device} isActive={true} frameProcessor={frameProcessor} frameProcessorFps={15} />
      <Text>PPG: {ppg}</Text>
    </SafeAreaView>
  );
}

6) On the JS side, post numeric PPG values to backend using fetch POST to /api/ppg/upload-ppg-value.

Notes:
- The native frame-processor must be implemented per vision-camera docs (the JS alone is not sufficient).
- This approach minimizes bandwidth and allows per-frame exposure control on iOS.
- I can help scaffold the exact plugin glue files for either iOS (Objective-C/Swift) or Android (Java/Kotlin) if you want to proceed.
