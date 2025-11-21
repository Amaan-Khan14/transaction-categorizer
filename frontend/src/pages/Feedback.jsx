import { useState, useEffect } from 'react';
import { getFeedbackStats } from '../services/api';
import './Feedback.css';

function Feedback() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchFeedbackStats();
  }, []);

  const fetchFeedbackStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getFeedbackStats();
      setStats(data);
    } catch (err) {
      setError('Failed to load feedback statistics. Please try again.');
      console.error('Error fetching feedback stats:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="feedback">
        <div className="feedback-header">
          <h2>Feedback Loop</h2>
          <p className="feedback-subtitle">
            Review and correct low-confidence predictions
          </p>
        </div>
        <div className="loading-container card">
          <div className="spinner"></div>
          <p>Loading feedback statistics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="feedback">
        <div className="feedback-header">
          <h2>Feedback Loop</h2>
          <p className="feedback-subtitle">
            Review and correct low-confidence predictions
          </p>
        </div>
        <div className="alert alert-error card">
          <strong>Error:</strong> {error}
        </div>
      </div>
    );
  }

  return (
    <div className="feedback">
      <div className="feedback-header">
        <h2>Feedback Loop</h2>
        <p className="feedback-subtitle">
          Review and correct low-confidence predictions to improve model accuracy
        </p>
      </div>

      <div className="feedback-container">
        {/* Overall Statistics */}
        <div className="stats-grid">
          <div className="stat-card card">
            <div className="stat-label">Total Feedback</div>
            <div className="stat-value">{stats?.total_feedback || 0}</div>
            <div className="stat-description">Total corrections submitted</div>
          </div>

          <div className="stat-card card">
            <div className="stat-label">Corrections Rate</div>
            <div className="stat-value">
              {stats?.total_feedback > 0
                ? `${((stats.total_feedback / (stats.total_feedback + 100)) * 100).toFixed(1)}%`
                : '0%'}
            </div>
            <div className="stat-description">Prediction correction rate</div>
          </div>

          <div className="stat-card card">
            <div className="stat-label">Active Learning</div>
            <div className="stat-value">Enabled</div>
            <div className="stat-description">Continuous model improvement</div>
          </div>

          <div className="stat-card card">
            <div className="stat-label">Feedback Items</div>
            <div className="stat-value">
              {stats?.recent_feedback?.length || 0}
            </div>
            <div className="stat-description">Recent corrections logged</div>
          </div>
        </div>

        {/* Category Corrections */}
        {stats?.corrections_by_category && Object.keys(stats.corrections_by_category).length > 0 && (
          <div className="corrections-section card">
            <h3>Category Corrections Breakdown</h3>
            <p className="section-subtitle">
              Common patterns in prediction corrections
            </p>
            <div className="corrections-grid">
              {Object.entries(stats.corrections_by_category)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 10)
                .map(([category, count], index) => (
                  <div key={index} className="correction-item">
                    <div className="correction-header">
                      <span className="correction-category">{category}</span>
                      <span className="correction-count">{count}</span>
                    </div>
                    <div className="correction-bar">
                      <div
                        className="correction-fill"
                        style={{
                          width: `${(count / Math.max(...Object.values(stats.corrections_by_category))) * 100}%`,
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Recent Feedback Activity */}
        {stats?.recent_feedback && stats.recent_feedback.length > 0 && (
          <div className="recent-section card">
            <h3>Recent Feedback Activity</h3>
            <p className="section-subtitle">
              Latest corrections submitted to improve the model
            </p>
            <div className="recent-list">
              {stats.recent_feedback.slice(0, 10).map((item, index) => (
                <div key={index} className="recent-item">
                  <div className="recent-transaction">{item.transaction}</div>
                  <div className="recent-correction">
                    <span className="recent-predicted">{item.predicted_category}</span>
                    <span className="recent-arrow">→</span>
                    <span className="recent-correct">{item.correct_category}</span>
                  </div>
                  <div className="recent-confidence">
                    {(item.confidence * 100).toFixed(1)}% confidence
                  </div>
                  <div className="recent-timestamp">
                    {new Date(item.timestamp).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* No Feedback Message */}
        {stats?.total_feedback === 0 && (
          <div className="no-feedback card">
            <h3>No Feedback Yet</h3>
            <p>
              Start improving the model by submitting corrections on the Single Prediction page.
              When you encounter incorrect predictions, use the "Submit Correction" button to help
              the model learn and improve over time.
            </p>
            <div className="feedback-benefits">
              <h4>Benefits of Feedback:</h4>
              <ul>
                <li>Improve model accuracy for future predictions</li>
                <li>Identify common misclassification patterns</li>
                <li>Enable continuous learning and adaptation</li>
                <li>Build a better understanding of transaction categories</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Feedback;
