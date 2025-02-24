import React, {useState} from 'react'
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import {StockContext} from './StockContext'
import {useContext} from 'react'
import './App.css'
import habitatBackground from './assets/dino images/backgroundResized.png'
import trex from './assets/dino images/rexResized_with_label.png'
import pterodactyl from './assets/dino images/pteroResized_with_label.png'
import velociraptor from './assets/dino images/raptorResized_with_label.png'
import stegosaurus from './assets/dino images/stegoResized_with_label_THICK_ish.png'

function App() {
  const {inputValue, setInputValue} = useContext(StockContext);
  const navigate = useNavigate();

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  }

  const handleButtonClick = (tab) => {
    navigate(`/${tab}?stock=${encodeURIComponent(inputValue)}`);
  } 
  
  return (
    <>
      <div className = "container">
        <h1 className = "header">
          STOCK - O - SAURUS
        </h1>

        <p className = "subheader">
          The #1 dino-inspired stock analysis tool
        </p>

        <div className = "input-container">
            <input 
              type = "text" 
              className = "input-box" 
              placeholder = "NVDA, AMZN, etc."
              value = {inputValue}
              onChange = {handleInputChange}
            ></input>
        </div>

        <div className = "dino-habitat">
          <img src = {habitatBackground}></img>

          <button 
            className = "rex-style"
            onClick={() => handleButtonClick("return-rex")}>
              <img src = {trex}></img>
          </button>

          <button 
            className = "ptero-style"
            onClick={() => handleButtonClick("trendodactyl")}>
              <img src = {pterodactyl}></img>
          </button>

          <button 
            className = "stego-style"
            onClick={() => handleButtonClick("stock-o-saurus")}>
              <img src = {stegosaurus}></img>
          </button>

          <button 
            className = "raptor-style"
            onClick={() => handleButtonClick("volatiliraptor")}>
              <img src = {velociraptor}></img>
          </button>
        </div>
      </div>
    </>
  )
}

export default App


/*
to do
- import Jurrasic Park font correctly
- move placeholder text above input box
- more animations in general
- the names of the functions for the dinosaurs (write them)
*/