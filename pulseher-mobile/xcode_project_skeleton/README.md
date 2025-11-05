Xcode Project Skeleton for PulseHER PPG

This folder contains Swift example files and instructions to incorporate them into an Xcode project.

Steps to create a runnable app in Xcode:
1. Open Xcode -> File -> New -> Project -> App (iOS) -> Next
2. Enter Product Name e.g., PulseHERNative, select Interface: Storyboard or SwiftUI, Language: Swift.
3. Save the project and open the generated folder in Finder.
4. Copy the files from this folder (AVCapturePPG.swift and ViewController_PPG.swift) into your Xcode project's source folder.
5. In Info.plist add NSCameraUsageDescription with a user-facing message.
6. Connect UI: If using Storyboards, set the initial ViewController class to ViewController_PPG; or instantiate it in SceneDelegate/AppDelegate if using SwiftUI.
7. Build & Run on a real device (required for torch). Test PPG values displayed and network POSTs.

Files included in this folder:
- AVCapturePPG.swift (video capture and green-channel extraction)
- ViewController_PPG.swift (UI + posting numeric PPG values)

Notes:
- You must run on a real device to use torch. Simulators do not support camera access.
- Replace the backend IP in ViewController_PPG.swift with your machine LAN IP.
- For production, handle privacy/medical disclaimers and follow App Store policies.
