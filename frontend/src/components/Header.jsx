import { useState, useEffect } from 'react';
import './Header.css';

function Header({ toggleSidebar, sidebarCollapsed }) {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };

  return (
    <header className="header">
      <div className="header-content">
        <div className="header-left">
          <button onClick={toggleSidebar} className="sidebar-toggle" aria-label="Toggle sidebar">
            <span className="toggle-icon">{sidebarCollapsed ? '☰' : '×'}</span>
          </button>
          <h1 className="header-title">Transaction Categorization</h1>
          <span className="header-subtitle">AI-Powered Financial Classification</span>
        </div>
        <div className="header-right">
          <button onClick={toggleTheme} className="theme-toggle" aria-label="Toggle theme">
            <span className="toggle-icon">{theme === 'light' ? '◐' : '◑'}</span>
          </button>
        </div>
      </div>
    </header>
  );
}

export default Header;
