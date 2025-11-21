import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  ArcElement,
  PointElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar, Doughnut, Line } from 'react-chartjs-2';
import './Insights.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  ArcElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE_URL = 'http://localhost:8000/api/v1';

function Insights() {
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [demoDataGenerated, setDemoDataGenerated] = useState(false);
  const [generatingData, setGeneratingData] = useState(false);

  // Generate demo data on mount if not already generated
  useEffect(() => {
    checkDemoData();
  }, []);

  const checkDemoData = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/insights/demo-transactions`);
      if (response.data.total > 0) {
        setDemoDataGenerated(true);
        // Automatically analyze demo data
        analyzeDemoData();
      }
    } catch (error) {
      // No demo data, that's fine
      console.log('No demo data available yet');
    }
  };

  const generateDemoData = async () => {
    setGeneratingData(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/insights/generate-demo-data`, {
        num_transactions: 150,
        days_back: 90,
        seed: 42
      });

      console.log('Demo data generated:', response.data);
      setDemoDataGenerated(true);

      // Automatically analyze after generation
      await analyzeDemoData();
    } catch (error) {
      console.error('Error generating demo data:', error);
      setError('Failed to generate demo data: ' + (error.response?.data?.detail || error.message));
    } finally {
      setGeneratingData(false);
    }
  };

  const analyzeDemoData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_BASE_URL}/insights/analyze-demo`);
      setInsights(response.data);
      console.log('Insights generated:', response.data);
    } catch (error) {
      console.error('Error analyzing data:', error);
      setError('Failed to analyze data: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  // Chart: Spending by Category
  const getCategoryChartData = () => {
    if (!insights || !insights.spending_by_category) return null;

    const sortedData = [...insights.spending_by_category].sort((a, b) => b.total_amount - a.total_amount);

    return {
      labels: sortedData.map(item => item.category),
      datasets: [{
        label: 'Total Spending (₹)',
        data: sortedData.map(item => item.total_amount),
        backgroundColor: [
          '#2563EB', '#10B981', '#F59E0B', '#EF4444',
          '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'
        ],
        borderColor: '#fff',
        borderWidth: 2
      }]
    };
  };

  // Chart: Spending by Day of Week
  const getDayChartData = () => {
    if (!insights || !insights.spending_by_day) return null;

    return {
      labels: insights.spending_by_day.map(item => item.day),
      datasets: [{
        label: 'Average Spending (₹)',
        data: insights.spending_by_day.map(item => item.average_spending),
        backgroundColor: 'rgba(37, 99, 235, 0.7)',
        borderColor: '#2563EB',
        borderWidth: 2
      }]
    };
  };

  // Chart: Category Distribution (Doughnut)
  const getCategoryDistributionData = () => {
    if (!insights || !insights.spending_by_category) return null;

    return {
      labels: insights.spending_by_category.map(item => item.category),
      datasets: [{
        data: insights.spending_by_category.map(item => item.percentage_of_total),
        backgroundColor: [
          '#2563EB', '#10B981', '#F59E0B', '#EF4444',
          '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'
        ],
        borderColor: '#fff',
        borderWidth: 3
      }]
    };
  };

  // Chart: Forecast
  const getForecastChartData = () => {
    if (!insights || !insights.forecasts) return null;

    return {
      labels: insights.forecasts.map(f => f.category),
      datasets: [
        {
          label: 'Current Month (₹)',
          data: insights.forecasts.map(f => f.current_month_spending),
          backgroundColor: 'rgba(37, 99, 235, 0.7)',
          borderColor: '#2563EB',
          borderWidth: 2
        },
        {
          label: 'Forecasted Next Month (₹)',
          data: insights.forecasts.map(f => f.forecasted_next_month),
          backgroundColor: 'rgba(16, 185, 129, 0.7)',
          borderColor: '#10B981',
          borderWidth: 2
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top'
      }
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right'
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return context.label + ': ' + context.parsed.toFixed(1) + '%';
          }
        }
      }
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'HIGH': return '#EF4444';
      case 'MEDIUM': return '#F59E0B';
      case 'LOW': return '#10B981';
      default: return '#6B7280';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'HIGH': return '#EF4444';
      case 'MEDIUM': return '#F59E0B';
      case 'LOW': return '#FCD34D';
      default: return '#6B7280';
    }
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'savings': return 'SAVE';
      case 'alert': return 'ALERT';
      case 'optimization': return 'OPT';
      case 'info': return 'INFO';
      default: return 'DATA';
    }
  };

  if (!demoDataGenerated && !generatingData) {
    return (
      <div className="insights-container">
        <div className="insights-header">
          <h1>AI-Powered Financial Insights</h1>
          <p>Generate sample transaction data to see intelligent insights</p>
        </div>

        <div className="empty-state">
          <h2>No Transaction Data Available</h2>
          <p>To see AI-powered insights, we need some transaction history with dates and amounts.</p>
          <button className="btn-primary" onClick={generateDemoData}>
            Generate Demo Data (150 transactions)
          </button>
          <p className="help-text">
            This will create 90 days of realistic transaction history for demo purposes
          </p>
        </div>
      </div>
    );
  }

  if (generatingData || loading) {
    return (
      <div className="insights-container">
        <div className="loading-state">
          <div className="spinner"></div>
          <p>{generatingData ? 'Generating transaction data...' : 'Analyzing transactions...'}</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="insights-container">
        <div className="error-state">
          <h2>Error</h2>
          <p>{error}</p>
          <button className="btn-primary" onClick={generateDemoData}>
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!insights) {
    return (
      <div className="insights-container">
        <div className="insights-header">
          <h1>AI-Powered Financial Insights</h1>
          <button className="btn-primary" onClick={analyzeDemoData}>
            Analyze Transactions
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="insights-container">
      <div className="insights-header">
        <h1>AI-Powered Financial Insights</h1>
        <div className="header-actions">
          <button className="btn-secondary" onClick={generateDemoData} disabled={generatingData}>
            {generatingData ? 'Generating...' : 'Regenerate Data'}
          </button>
          <button className="btn-primary" onClick={analyzeDemoData} disabled={loading}>
            {loading ? 'Analyzing...' : 'Refresh Analysis'}
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="summary-grid">
        <div className="summary-card">
          <div className="summary-content">
            <div className="summary-label">Total Spending</div>
            <div className="summary-value">₹{insights.total_spending.toFixed(2)}</div>
          </div>
        </div>

        <div className="summary-card">
          <div className="summary-content">
            <div className="summary-label">Total Transactions</div>
            <div className="summary-value">{insights.total_transactions}</div>
          </div>
        </div>

        <div className="summary-card anomaly">
          <div className="summary-content">
            <div className="summary-label">Anomalies Detected</div>
            <div className="summary-value">{insights.anomalies.length}</div>
          </div>
        </div>

        <div className="summary-card recommendation">
          <div className="summary-content">
            <div className="summary-label">Recommendations</div>
            <div className="summary-value">{insights.recommendations.length}</div>
          </div>
        </div>
      </div>

      {/* Anomalies Section */}
      {insights.anomalies && insights.anomalies.length > 0 && (
        <div className="insights-section">
          <h2>Unusual Transactions Detected</h2>
          <div className="anomalies-grid">
            {insights.anomalies.slice(0, 6).map((anomaly, idx) => (
              <div key={idx} className="anomaly-card">
                <div className="anomaly-header">
                  <span className="anomaly-badge" style={{ backgroundColor: getSeverityColor(anomaly.severity) }}>
                    {anomaly.severity}
                  </span>
                  <span className="anomaly-amount">₹{anomaly.amount.toFixed(2)}</span>
                </div>
                <div className="anomaly-transaction">{anomaly.transaction}</div>
                <div className="anomaly-category">{anomaly.category}</div>
                <div className="anomaly-reason">{anomaly.reason}</div>
                <div className="anomaly-date">{anomaly.date}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations Section */}
      {insights.recommendations && insights.recommendations.length > 0 && (
        <div className="insights-section">
          <h2>Smart Recommendations</h2>
          <div className="recommendations-grid">
            {insights.recommendations.map((rec, idx) => (
              <div key={idx} className="recommendation-card">
                <div className="recommendation-header">
                  <span className="recommendation-type-badge" style={{ backgroundColor: getPriorityColor(rec.priority) }}>
                    {getTypeIcon(rec.type)}
                  </span>
                  <span className="recommendation-priority" style={{ color: getPriorityColor(rec.priority) }}>
                    {rec.priority} PRIORITY
                  </span>
                </div>
                <h3>{rec.title}</h3>
                <p>{rec.message}</p>
                {rec.potential_savings && (
                  <div className="savings-badge">
                    Potential Savings: ₹{rec.potential_savings.toFixed(0)}/month
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Charts Section */}
      <div className="charts-section">
        <div className="chart-grid">
          {/* Spending by Category */}
          <div className="chart-card">
            <h3>Spending by Category</h3>
            <div className="chart-wrapper">
              {getCategoryChartData() && (
                <Bar data={getCategoryChartData()} options={chartOptions} />
              )}
            </div>
          </div>

          {/* Category Distribution */}
          <div className="chart-card">
            <h3>Category Distribution</h3>
            <div className="chart-wrapper">
              {getCategoryDistributionData() && (
                <Doughnut data={getCategoryDistributionData()} options={doughnutOptions} />
              )}
            </div>
          </div>

          {/* Spending by Day of Week */}
          <div className="chart-card full-width">
            <h3>Spending Patterns by Day of Week</h3>
            <div className="chart-wrapper">
              {getDayChartData() && (
                <Bar data={getDayChartData()} options={chartOptions} />
              )}
            </div>
          </div>

          {/* Forecast */}
          <div className="chart-card full-width">
            <h3>Spending Forecast: Current vs. Next Month</h3>
            <div className="chart-wrapper">
              {getForecastChartData() && (
                <Bar data={getForecastChartData()} options={chartOptions} />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Analysis Period */}
      <div className="analysis-footer">
        <p>
          Analysis Period: {insights.analysis_period.start} to {insights.analysis_period.end}
        </p>
      </div>
    </div>
  );
}

export default Insights;
