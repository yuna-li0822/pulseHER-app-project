// AVCapturePPG.swift
// Simple native iOS example showing how to capture frames with AVFoundation
// and compute a mean green-channel value per frame. This is a developer sample
// (not a complete app). Use from a ViewController or SwiftUI App with proper
// permissions and Info.plist entries (NSCameraUsageDescription).

import AVFoundation
import UIKit

class AVCapturePPG: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "ppg.capture.queue")
    var onPPGValue: ((Double) -> Void)?

    func start() {
        session.beginConfiguration()
        session.sessionPreset = .vga640x480

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("Camera not available")
            return
        }

        if session.canAddInput(input) { session.addInput(input) }

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }

        // Try to lock exposure and set torch if available
        do {
            try device.lockForConfiguration()
            if device.isExposureModeSupported(.locked) {
                device.exposureMode = .locked
            }
            if device.hasTorch {
                try device.setTorchModeOn(level: 1.0)
            }
            device.unlockForConfiguration()
        } catch {
            print("Could not configure device: \(error)")
        }

        session.commitConfiguration()
        session.startRunning()
    }

    func stop() {
        session.stopRunning()
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back), device.hasTorch {
            do { try device.lockForConfiguration(); device.torchMode = .off; device.unlockForConfiguration() } catch {}
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags.readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags.readOnly) }

        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return }

        let bufferPtr = baseAddress.assumingMemoryBound(to: UInt8.self)
        var greenTotal: UInt64 = 0
        var pixels: UInt64 = 0

        for y in 0..<height {
            let row = bufferPtr + y * bytesPerRow
            for x in 0..<width {
                let pixel = row + x * 4
                let b = pixel[0]
                let g = pixel[1]
                let r = pixel[2]
                // BGRA ordering
                greenTotal += UInt64(g)
                pixels += 1
            }
        }

        if pixels > 0 {
            let meanGreen = Double(greenTotal) / Double(pixels)
            // Normalize to 0..1 (assuming 0..255)
            let normalized = meanGreen / 255.0
            DispatchQueue.main.async {
                self.onPPGValue?(normalized)
            }
        }
    }
}
