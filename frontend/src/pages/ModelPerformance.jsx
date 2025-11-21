import { useState, useEffect } from 'react';
import { getMetrics } from '../services/api';
import './ModelPerformance.css';

function ModelPerformance() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const data = await getMetrics();
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError('Failed to load metrics');
      console.error('Error fetching metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="model-performance">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading performance metrics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="model-performance">
        <div className="alert alert-error">{error}</div>
      </div>
    );
  }

  const overall_metrics = metrics?.overall_metrics || {};
  const per_category_metrics = metrics?.per_category_metrics || [];
  const usage_stats = metrics?.usage_stats || {};

  return (
    <div className="model-performance">
      <div className="performance-header">
        <h2>Model Performance</h2>
        <p className="performance-subtitle">
          Evaluated on {overall_metrics?.test_samples || 0} test samples across {overall_metrics?.num_categories || 0} categories
        </p>
      </div>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">F1 Score</div>
          <div className="metric-value primary">
            {overall_metrics?.f1_macro ? (overall_metrics.f1_macro * 100).toFixed(2) : 'N/A'}%
          </div>
          <div className="metric-description">Macro-averaged F1 score</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Accuracy</div>
          <div className="metric-value success">
            {overall_metrics?.accuracy ? (overall_metrics.accuracy * 100).toFixed(2) : 'N/A'}%
          </div>
          <div className="metric-description">Overall prediction accuracy</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Balanced Acc</div>
          <div className="metric-value primary">
            {overall_metrics?.balanced_accuracy ? (overall_metrics.balanced_accuracy * 100).toFixed(2) : 'N/A'}%
          </div>
          <div className="metric-description">Class-balanced accuracy</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Latency</div>
          <div className="metric-value latency">
            {usage_stats?.avg_latency_ms ? usage_stats.avg_latency_ms.toFixed(0) : 'N/A'}ms
          </div>
          <div className="metric-description">Average response time</div>
        </div>
      </div>

      <div className="category-metrics">
        <h3>Per-Category Performance</h3>
        <p className="text-gray">
          Detailed breakdown of precision, recall, and F1-score for each category
        </p>

        <div className="category-grid">
          {per_category_metrics.map((cat) => (
            <div key={cat.category} className="category-item">
              <div className="category-header">
                <span className="category-name">{cat.category}</span>
                <span className="category-f1">
                  {(cat['f1-score'] * 100).toFixed(2)}%
                </span>
              </div>

              <div className="category-metrics-row">
                <div className="metric-item">
                  <div className="metric-item-label">Precision</div>
                  <div className="metric-item-bar">
                    <div
                      className="metric-item-fill precision"
                      style={{ width: `${cat.precision * 100}%` }}
                    ></div>
                  </div>
                  <div className="metric-item-value">
                    {(cat.precision * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="metric-item">
                  <div className="metric-item-label">Recall</div>
                  <div className="metric-item-bar">
                    <div
                      className="metric-item-fill recall"
                      style={{ width: `${cat.recall * 100}%` }}
                    ></div>
                  </div>
                  <div className="metric-item-value">
                    {(cat.recall * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="metric-item">
                  <div className="metric-item-label">Support</div>
                  <div className="metric-item-bar">
                    <div
                      className="metric-item-fill support"
                      style={{ width: `${(cat.support / 200) * 100}%` }}
                    ></div>
                  </div>
                  <div className="metric-item-value">{cat.support} samples</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="additional-metrics">
        <h3>Additional Information</h3>
        <div className="additional-grid">
          <div className="additional-item">
            <span className="additional-label">F1 Micro</span>
            <span className="additional-value">
              {overall_metrics?.f1_micro ? (overall_metrics.f1_micro * 100).toFixed(2) : 'N/A'}%
            </span>
          </div>
          <div className="additional-item">
            <span className="additional-label">F1 Weighted</span>
            <span className="additional-value">
              {overall_metrics?.f1_weighted ? (overall_metrics.f1_weighted * 100).toFixed(2) : 'N/A'}%
            </span>
          </div>
          <div className="additional-item">
            <span className="additional-label">Categories</span>
            <span className="additional-value">{overall_metrics?.num_categories || 'N/A'}</span>
          </div>
          <div className="additional-item">
            <span className="additional-label">Model Version</span>
            <span className="additional-value">v1.0.0</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ModelPerformance;
