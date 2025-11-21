import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import ImagePredict from './pages/ImagePredict';
import BatchUpload from './pages/BatchUpload';
import ModelPerformance from './pages/ModelPerformance';
import Insights from './pages/Insights';
import Feedback from './pages/Feedback';
import './App.css';

function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  return (
    <Router>
      <div className="app">
        <Header toggleSidebar={toggleSidebar} sidebarCollapsed={sidebarCollapsed} />
        <div className="app-layout">
          <Sidebar collapsed={sidebarCollapsed} />
          <main className="main-content">
            <div className="content-wrapper">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/predict" element={<Predict />} />
                <Route path="/image" element={<ImagePredict />} />
                <Route path="/batch" element={<BatchUpload />} />
                <Route path="/insights" element={<Insights />} />
                <Route path="/performance" element={<ModelPerformance />} />
                <Route path="/feedback" element={<Feedback />} />
              </Routes>
            </div>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
