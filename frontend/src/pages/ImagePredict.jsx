import { useState } from "react";
import ConfidenceMeter from "../components/ConfidenceMeter";
import api from "../services/api";
import "./ImagePredict.css";

function ImagePredict() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        setError("Please select a valid image file");
        return;
      }
      setSelectedImage(file);
      setError(null);

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!selectedImage) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);

      const response = await api.post("/predict/image", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Failed to process image. Please try a clearer image."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="image-predict">
      <div className="image-predict-header">
        <h2>Receipt & Image Classification</h2>
        <p className="image-predict-subtitle">
          Upload a receipt, screenshot, or transaction image for instant categorization
        </p>
      </div>

      <div className="image-predict-container">
        {/* Upload Section */}
        <div className="upload-section card">
          <h3>Upload Image</h3>
          <p className="text-gray text-sm mb-4">
            Supports: JPEG, PNG, WebP (supports receipts, screenshots, invoices)
          </p>

          <form onSubmit={handlePredict}>
            {/* Image Preview or Upload Area */}
            <div
              className="image-upload-area"
              onClick={() => document.getElementById("image-input").click()}
            >
              {imagePreview ? (
                <div className="image-preview-container">
                  <img src={imagePreview} alt="Preview" className="preview-img" />
                  <div className="preview-overlay">Click to change</div>
                </div>
              ) : (
                <div className="upload-placeholder">
                  <svg
                    width="48"
                    height="48"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                  >
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  <p>Click to upload or drag and drop</p>
                  <p className="text-secondary">PNG, JPG, WebP up to 10MB</p>
                </div>
              )}
              <input
                id="image-input"
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                disabled={loading}
                style={{ display: "none" }}
              />
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !selectedImage}
              style={{ marginTop: "20px", width: "100%" }}
            >
              {loading ? "Analyzing Image..." : "Classify Image"}
            </button>
          </form>
        </div>

        {/* Error Message */}
        {error && (
          <div className="alert alert-error card">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="result-section card">
            <h3>Classification Result</h3>

            {/* Extracted Text */}
            {result.transaction && (
              <div className="extracted-text-box">
                <h4>Extracted Text</h4>
                <div className="extracted-text">
                  "{result.transaction}"
                </div>
              </div>
            )}

            {/* Main Result */}
            <div className="result-main">
              <div className="result-category">
                <span className="result-label">Predicted Category</span>
                <span className="result-value">{result.predicted_category}</span>
              </div>

              <ConfidenceMeter confidence={result.confidence} />
            </div>

            {/* Explainability */}
            {result.explanation && (
              <div className="explanation-section">
                <h4>Why This Category?</h4>

                {result.explanation.top_features &&
                  result.explanation.top_features.length > 0 && (
                    <div className="features-explanation">
                      <h5>Key Words Influencing This Prediction</h5>
                      <div className="features-list">
                        {result.explanation.top_features.map((feat, index) => (
                          <div
                            key={index}
                            className={`feature-item impact-${feat.impact?.replace("_", "-")}`}
                          >
                            <div className="feature-header">
                              <span className="feature-name">{feat.feature}</span>
                              <span className="feature-weight">
                                {feat.weight > 0 ? "+" : ""}
                                {feat.weight.toFixed(3)}
                              </span>
                            </div>
                            {feat.impact && (
                              <div className="feature-impact">
                                <span className="impact-label">
                                  {feat.impact.replace(/_/g, " ")}
                                </span>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                {result.explanation.model_votes &&
                  Object.keys(result.explanation.model_votes).length > 0 && (
                    <div className="model-votes">
                      <h5>Individual Model Predictions</h5>
                      <div className="votes-grid">
                        {Object.entries(result.explanation.model_votes).map(
                          ([modelName, vote], index) => (
                            <div key={index} className="vote-card">
                              <div className="model-name">{modelName}</div>
                              <div className="vote-category">{vote.category}</div>
                              <div className="vote-confidence">
                                {(vote.confidence * 100).toFixed(1)}%
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}
              </div>
            )}

            {/* Alternatives */}
            {result.alternatives && result.alternatives.length > 0 && (
              <div className="alternatives-section">
                <h4>Alternative Classifications</h4>
                <div className="alternatives-list">
                  {result.alternatives.map((alt, index) => (
                    <div key={index} className="alternative-item">
                      <span className="alternative-category">{alt.category}</span>
                      <div className="alternative-confidence-bar">
                        <div
                          className="alternative-confidence-fill"
                          style={{ width: `${alt.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="alternative-percentage">
                        {(alt.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Metadata */}
            <div className="result-metadata">
              <div className="metadata-item">
                <span>Processing Time</span>
                <span>{result.metadata.processing_time_ms.toFixed(2)}ms</span>
              </div>
              <div className="metadata-item">
                <span>Model Version</span>
                <span>{result.metadata.model_version}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ImagePredict;
