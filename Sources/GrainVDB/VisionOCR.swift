import Foundation
import Vision
#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

/// Result container for detected UI text bounding boxes
public struct DetectedTextElement: Codable, Sendable {
    public let text: String
    public let confidence: Float
    public let x: Float
    public let y: Float
    public let width: Float
    public let height: Float

    public init(text: String, confidence: Float, x: Float, y: Float, width: Float, height: Float) {
        self.text = text
        self.confidence = confidence
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

/// Native Apple Silicon Vision Framework OCR engine
public final class AppleVisionOCR: @unchecked Sendable {
    public init() {}

    /// Extracts all text elements and normalized bounding boxes from image data
    public func recognizeText(from imageData: Data, fastMode: Bool = true) throws -> [DetectedTextElement] {
        guard let imageSource = CGImageSourceCreateWithData(imageData as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            return []
        }
        return try recognizeText(from: cgImage, fastMode: fastMode)
    }

    /// Extracts all text elements directly from a CGImage in memory
    public func recognizeText(from cgImage: CGImage, fastMode: Bool = true) throws -> [DetectedTextElement] {
        var results: [DetectedTextElement] = []
        
        let request = VNRecognizeTextRequest { request, error in
            guard let observations = request.results as? [VNRecognizedTextObservation] else {
                return
            }
            for obs in observations {
                guard let candidate = obs.topCandidates(1).first else { continue }
                let box = obs.boundingBox
                let element = DetectedTextElement(
                    text: candidate.string,
                    confidence: candidate.confidence,
                    x: Float(box.origin.x),
                    y: Float(box.origin.y),
                    width: Float(box.size.width),
                    height: Float(box.size.height)
                )
                results.append(element)
            }
        }

        request.recognitionLevel = fastMode ? .fast : .accurate
        request.usesLanguageCorrection = false // Faster for UI element tokens

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        return results
    }

    /// Convenience helper returning all detected text as a single space-separated string
    public func extractFullText(from imageData: Data) -> String {
        guard let elements = try? recognizeText(from: imageData) else { return "" }
        return elements.map { $0.text }.joined(separator: " ")
    }
}
