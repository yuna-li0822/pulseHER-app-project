// ViewController_PPG.swift
// Example ViewController integrating AVCapturePPG and posting values to backend

import UIKit

class ViewController_PPG: UIViewController {
    let ppg = AVCapturePPG()
    let apiURL = URL(string: "http://192.168.X.X:5000/api/ppg/upload-ppg-value")! // replace IP
    var valueLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        valueLabel = UILabel(frame: CGRect(x: 20, y: 80, width: 300, height: 40))
        valueLabel.text = "PPG: --"
        view.addSubview(valueLabel)

        ppg.onPPGValue = { [weak self] val in
            DispatchQueue.main.async {
                self?.valueLabel.text = String(format: "PPG: %.3f", val)
            }
            // Post numeric value to backend
            var request = URLRequest(url: self!.apiURL)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let body = ["ppg_value": val]
            request.httpBody = try? JSONSerialization.data(withJSONObject: body, options: [])
            URLSession.shared.dataTask(with: request).resume()
        }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        ppg.start()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        ppg.stop()
    }
}
