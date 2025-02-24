import {useLocation} from 'react-router-dom';
import {useNavigate} from 'react-router-dom';
import stegosaurus from './assets/dino images/stegoResized.png'
import './StockOSaurus.css'
import graph from './assets/graph1.png'

function StockOSaurus () {
    let navigate = useNavigate();
    
    const location = useLocation(); // use location for passing data in btwn pages
    const queryParams = new URLSearchParams(location.search);
    const stockToken = queryParams.get('stock')

    const handleButtonClick = () => {
        navigate(`/?stock=${encodeURIComponent(stockToken)}`);
    }
    
    return (
        <>
            <img src = {stegosaurus} className = "dino-stego"></img>
            <div class = "triangle-container">
                <div class = "container-rex">
                    <h1 className = "header">Stock - O - Saurus</h1>
                    
                    <div className = "graphBox">
                        <img src = {graph} className = "graph"></img>
                        <p className = "big-idea">Excellent return potential (Rating 100)</p>
                    </div>

                    <div className = "analysis-box">
                        <p>indicator1:<br/>(value)</p>
                        <p>description of big idea</p>
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

export default StockOSaurus