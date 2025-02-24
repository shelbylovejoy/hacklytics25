import {useLocation} from 'react-router-dom';
import {useNavigate} from 'react-router-dom';
import pterodactyl from './assets/dino images/pteroResized.png'
import './Trendodactyl.css'
import graph from './assets/graph1.png'

function Trendodactyl () {
    let navigate = useNavigate();
    
    const location = useLocation(); // use location for passing data in btwn pages
    const queryParams = new URLSearchParams(location.search);
    const stockToken = queryParams.get('stock')

    const handleButtonClick = () => {
        navigate(`/?stock=${encodeURIComponent(stockToken)}`);
    }
    
    return (
        <>
            <img src = {pterodactyl} className = "dino-ptero"></img>
            <div class = "triangle-container">
                <div class = "container-rex">
                    <h1 className = "header">TREND-O-DACTYL</h1>

                    <div className = "graphBox">
                        <img src = {graph} className = "graph"></img>
                        <p className = "big-idea">Excellent return potential (Rating 100)</p>
                    </div>

                    <div className = "analysis-box">
                        <p>indicator1:<br/>(value)</p>
                        <p className = "middle">Assesses price momentum using linear regression slope, RSI, and MACD. Higher rating indicates stronger upward trend.</p>
                        <p>indicator2:<br/>(value)</p>
                    </div>
                    
                    <button
                        className = "back-button" 
                        onClick={() => handleButtonClick()}>
                        RETURN TO HABITAT
                    </button>
                </div>
            </div>
        </>
    )
}

export default Trendodactyl