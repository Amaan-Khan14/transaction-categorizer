import './ConfidenceMeter.css';

function ConfidenceMeter({ confidence }) {
  const percentage = (confidence * 100).toFixed(1);

  return (
    <div className="confidence-meter">
      <div className="confidence-header">
        <span className="confidence-label">Confidence Score</span>
      </div>

      <div className="confidence-value">
        {percentage}%
      </div>

      <div className="confidence-bar">
        <div
          className="confidence-fill"
          style={{ width: `${percentage}%` }}
        ></div>
      </div>

      <div className="confidence-scale">
        <span>0%</span>
        <span className="confidence-marker" style={{ left: '70%' }}>
          70%
        </span>
        <span className="confidence-marker" style={{ left: '85%' }}>
          85%
        </span>
        <span>100%</span>
      </div>
    </div>
  );
}

export default ConfidenceMeter;
