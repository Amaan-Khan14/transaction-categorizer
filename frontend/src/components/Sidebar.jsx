import { Link, useLocation } from 'react-router-dom';
import './Sidebar.css';

function Sidebar({ collapsed }) {
  const location = useLocation();

  const menuItems = [
    { path: '/', label: 'Dashboard' },
    { path: '/predict', label: 'Single Prediction' },
    { path: '/image', label: 'Image Classification' },
    { path: '/batch', label: 'Batch Upload' },
    { path: '/insights', label: 'AI Insights' },
    { path: '/performance', label: 'Model Performance' },
    { path: '/feedback', label: 'Feedback' },
  ];

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <nav className="sidebar-nav">
        {menuItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`sidebar-item ${location.pathname === item.path ? 'active' : ''}`}
            title={collapsed ? item.label : ''}
          >
            <span className="sidebar-label">{item.label}</span>
          </Link>
        ))}
      </nav>
      {!collapsed && (
        <div className="sidebar-footer">
          <div className="sidebar-info">
            <p className="sidebar-info-title">Model Version</p>
            <p className="sidebar-info-value">v1.0.0</p>
          </div>
          <div className="sidebar-info">
            <p className="sidebar-info-title">F1 Score</p>
            <p className="sidebar-info-value">96.15%</p>
          </div>
        </div>
      )}
    </aside>
  );
}

export default Sidebar;
