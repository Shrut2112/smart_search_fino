import React from 'react'
import ReactDOM from 'react-dom/client'
import './i18n'       // must be imported before App
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)