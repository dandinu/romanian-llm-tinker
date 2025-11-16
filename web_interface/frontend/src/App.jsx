import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { Training } from './pages/Training'
import { Datasets } from './pages/Datasets'
import { Testing } from './pages/Testing'
import { Settings } from './pages/Settings'

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/training" element={<Training />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/testing" element={<Testing />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App
