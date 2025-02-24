import {useLocation} from 'react-router-dom';
import {useNavigate} from 'react-router-dom';
import trex from './assets/dino images/rexResized.png'
import tree from './assets/dino images/treeResized.png'
import './ReturnRex.css'
import graph from './assets/graph1.png'

function ReturnRex () {
    let navigate = useNavigate();
    
    const location = useLocation(); // use location for passing data in btwn pages
    const queryParams = new URLSearchParams(location.search);
    const stockToken = queryParams.get('stock')

    const handleButtonClick = () => {
        navigate(`/?stock=${encodeURIComponent(stockToken)}`);
    }
    
    return (
        <>
            <img src = {trex} className = "dino-rex"></img>
            <div class = "triangle-container">
                <div class = "container-rex">
                    <h1 className = "header">Return - Rex</h1>
                    <div className = "graphBox">
                        <img src = {graph} className = "graph"></img>
                        <p className = "big-idea">Excellent return potential (Rating 100)</p>
                    </div>

                    <div className = "analysis-box">
                        <p>indicator1:<br/>(value)</p>
                        <p className = "middle">Reflects profitability potential using CAGR, moving averages, and beta. Higher rating indicates stronger return prospects.</p>
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

export default ReturnRex