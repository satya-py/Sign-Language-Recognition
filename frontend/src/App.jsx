import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import About from './pages/About';
import Detection from './pages/Detection';
import Signs from './pages/Signs';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background text-foreground flex flex-col">
        <Navbar />
        <main className="flex-grow pt-16">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/detection" element={<Detection />} />
            <Route path="/signs" element={<Signs />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
