import { useState } from 'react';
import { predictSingle, submitFeedback, getTaxonomy } from '../services/api';
import ConfidenceMeter from '../components/ConfidenceMeter';
import './Predict.css';

function Predict() {
  const [transaction, setTransaction] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [feedbackSuccess, setFeedbackSuccess] = useState(false);

  const exampleTransactions = [
    'BigBasket grocery delivery',
    'Myntra fashion store',
    'Swiggy food order',
    'Ola cab ride to airport',
    'JioMart monthly groceries',
    'Hotstar premium subscription',
  ];

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!transaction.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setShowFeedbackForm(false);
    setFeedbackSuccess(false);

    try {
      const data = await predictSingle(transaction);
      setResult(data);

      // Fetch categories for feedback
      try {
        const taxonomyData = await getTaxonomy();
        setCategories(Array.isArray(taxonomyData.categories) ? taxonomyData.categories : []);
      } catch (taxonomyErr) {
        console.error('Failed to fetch taxonomy:', taxonomyErr);
        // Set empty array if taxonomy fetch fails - feedback form will still work
        setCategories([]);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitFeedback = async () => {
    if (!selectedCategory) return;

    setFeedbackLoading(true);
    setError(null);

    try {
      await submitFeedback({
        transaction: transaction,
        predicted_category: result.predicted_category,
        correct_category: selectedCategory,
        confidence: result.confidence,
        user_id: null
      });
      setFeedbackSuccess(true);
      setShowFeedbackForm(false);
      setTimeout(() => setFeedbackSuccess(false), 3000);
    } catch (err) {
      console.error('Feedback submission error:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to submit feedback. Please try again.';
      setError(errorMessage);
      setFeedbackSuccess(false);
    } finally {
      setFeedbackLoading(false);
    }
  };

  const useExample = (example) => {
    setTransaction(example);
  };

  return (
    <div className="predict">
      <div className="predict-header">
        <h2>Single Transaction Prediction</h2>
        <p className="predict-subtitle">
          Enter a transaction description to get AI-powered categorization
        </p>
      </div>

      <div className="predict-container">
        <div className="predict-form-section card">
          <form onSubmit={handlePredict}>
            <label htmlFor="transaction">Transaction Description</label>
            <input
              id="transaction"
              type="text"
              value={transaction}
              onChange={(e) => setTransaction(e.target.value)}
              placeholder="e.g., BigBasket grocery delivery"
              className="predict-input"
              disabled={loading}
            />

            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !transaction.trim()}
            >
              {loading ? 'Analyzing...' : 'Predict Category'}
            </button>
          </form>

          <div className="examples-section">
            <p className="examples-label">Try these examples:</p>
            <div className="examples-grid">
              {exampleTransactions.map((example, index) => (
                <button
                  key={index}
                  onClick={() => useExample(example)}
                  className="btn btn-secondary example-btn"
                  disabled={loading}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        </div>

        {error && (
          <div className="alert alert-error card">
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="result-section card">
            <h3>Prediction Result</h3>

            <div className="result-main">
              <div className="result-category">
                <span className="result-label">Predicted Category</span>
                <span className="result-value">{result.predicted_category}</span>
              </div>

              <ConfidenceMeter confidence={result.confidence} />
            </div>

            {result.explanation && (
              <div className="explanation-section">
                <h4>Explainability</h4>

                {result.explanation.top_features && result.explanation.top_features.length > 0 && (
                  <div className="features-explanation">
                    <h5>Key Features Influencing This Prediction</h5>
                    <div className="features-list">
                      {result.explanation.top_features.map((feat, index) => (
                        <div key={index} className={`feature-item impact-${feat.impact?.replace('_', '-')}`}>
                          <div className="feature-header">
                            <span className="feature-name">{feat.feature}</span>
                            <span className="feature-weight">{feat.weight > 0 ? '+' : ''}{feat.weight.toFixed(3)}</span>
                          </div>
                          {feat.impact && (
                            <div className="feature-impact">
                              <span className="impact-label">{feat.impact.replace(/_/g, ' ')}</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {result.explanation.model_votes && Object.keys(result.explanation.model_votes).length > 0 && (
                  <div className="model-votes">
                    <h5>Individual Model Predictions</h5>
                    <div className="votes-grid">
                      {Object.entries(result.explanation.model_votes).map(([modelName, vote], index) => (
                        <div key={index} className="vote-card">
                          <div className="model-name">{modelName}</div>
                          <div className="vote-category">{vote.category}</div>
                          <div className="vote-confidence">{(vote.confidence * 100).toFixed(1)}%</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {result.alternatives && result.alternatives.length > 0 && (
              <div className="alternatives-section">
                <h4>Alternative Predictions</h4>
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

            {feedbackSuccess && (
              <div className="feedback-success">
                Feedback submitted successfully! Thank you for helping improve the model.
              </div>
            )}

            {!showFeedbackForm ? (
              <div className="feedback-prompt">
                <p>Is this prediction incorrect?</p>
                <button
                  className="btn btn-secondary"
                  onClick={() => setShowFeedbackForm(true)}
                >
                  Submit Correction
                </button>
              </div>
            ) : (
              <div className="feedback-form">
                <h4>Correct the Prediction</h4>
                <p>Select the correct category for this transaction:</p>
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="feedback-select"
                >
                  <option value="">-- Select Correct Category --</option>
                  {categories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                </select>
                <div className="feedback-actions">
                  <button
                    className="btn btn-primary"
                    onClick={handleSubmitFeedback}
                    disabled={!selectedCategory || feedbackLoading}
                  >
                    {feedbackLoading ? 'Submitting...' : 'Submit Feedback'}
                  </button>
                  <button
                    className="btn btn-secondary"
                    onClick={() => setShowFeedbackForm(false)}
                    disabled={feedbackLoading}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default Predict;
