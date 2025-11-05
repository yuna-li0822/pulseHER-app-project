// GreenFrameProcessor.js
// NOTE: This is illustrative JS pseudo-code for a vision-camera frame processor.
// The real implementation requires native frame processor registration (see vision-camera docs).

export function processFrameGetMeanGreen(buffer, width, height) {
  // buffer expected to be Uint8Array BGRA
  let sum = 0;
  let count = 0;
  for (let i = 0; i < buffer.length; i += 4) {
    const g = buffer[i+1];
    sum += g;
    count += 1;
  }
  const meanG = sum / (count || 1);
  return meanG / 255.0;
}

// In your VisionCamera frame processor, call this logic in native worklet and post numeric values to backend.
