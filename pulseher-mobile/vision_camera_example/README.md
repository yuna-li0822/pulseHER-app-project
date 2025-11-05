Vision Camera example (react-native-vision-camera)

This folder contains a sample outline and JS code showing how to use react-native-vision-camera
with a frame processor to compute the median green channel on-device and only post numeric
PPG values to your backend.

Important: This requires a native build (Xcode) and cannot run inside Expo Go. Use bare React Native
or a custom dev client.

Steps summary:
1. Create a React Native app (not Expo managed) or use Expo bare workflow.
2. Install vision-camera and frame-processor plugin:
   yarn add react-native-vision-camera
   yarn add @shopify/react-native-skia  # optional for drawing
   yarn add react-native-reanimated
   cd ios && pod install
3. Request camera permission and enable frame processor.

Sample frame-processor snippet (JS):

// GreenFrameProcessor.js
import { FrameProcessorPlugin } from 'react-native-vision-camera';

export const greenFrameProcessor = FrameProcessorPlugin('greenFrameProcessor', (frame) => {
  'worklet'
  const width = frame.width;
  const height = frame.height;
  const data = frame.getData(); // platform-specific
  // compute median or mean of green channel from BGRA
  let sum = 0;
  let count = 0;
  for (let i = 0; i < data.length; i += 4) {
    // BGRA ordering
    const g = data[i+1];
    sum += g;
    count += 1;
  }
  const meanG = sum / count;
  // return normalized value 0..1
  return meanG / 255.0;
});

// Use this processor in your camera component and send numeric values to backend instead of images.

Notes:
- You must implement the native frame processor glue code per the vision-camera docs.
- This approach minimizes bandwidth and gives low-latency access to the green channel.
- You can aggregate several frames or run a small circular buffer and compute AC/DC/SNR on-device.
