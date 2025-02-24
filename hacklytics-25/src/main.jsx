import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import './index.css'
import StockProvider from './StockContext.jsx'
import App from './App.jsx'
import ReturnRex from './ReturnRex.jsx'
import Volatiliraptor from './Volatiliraptor.jsx'
import StockOSaurus from './StockOSaurus.jsx'
import Trendodactyl from './Trendodactyl.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <StockProvider>
    <Router>
      <Routes>
        <Route path = '/' element = {<App/>}></Route>
        <Route path = '/return-rex' element = {<ReturnRex/>}></Route>
        <Route path = '/volatiliraptor' element = {<Volatiliraptor/>}></Route>
        <Route path = '/stock-o-saurus' element = {<StockOSaurus/>}></Route>
        <Route path = '/trendodactyl' element = {<Trendodactyl/>}></Route>
      </Routes>
    </Router>
    </StockProvider>
  </StrictMode>
)
