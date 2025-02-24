import {useLocation} from 'react-router-dom';
import {useNavigate} from 'react-router-dom';
import velociraptor from './assets/dino images/raptorResized.png'
import './Volatiliraptor.css'
import graph from './assets/graph1.png'

function Volatiliraptor () {
    let navigate = useNavigate();
    
    const location = useLocation(); // use location for passing data in btwn pages
    const queryParams = new URLSearchParams(location.search);
    const stockToken = queryParams.get('stock')

    const handleButtonClick = () => {
        navigate(`/?stock=${encodeURIComponent(stockToken)}`);
    }
    
    return (
        <>
            <img src = {velociraptor} className = "dino-raptor"></img>
            <div class = "triangle-container">
                <div class = "container-rex">
                    <h1 className = "header">VOLATILIRAPTOR</h1>
                    <div className = "graphBox">
                        <img src = {graph} className = "graph"></img>
                        <p className = "big-idea">Excellent return potential (Rating 100)</p>
                    </div>

                    <div className = "analysis-box">
                        <p>indicator1:<br/>(value)</p>
                        <p className = "middle">Evaluates volatility and potential losses using historical volatility, max drawdown, and VaR. Higher rating suggests lower risk.</p>
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

export default Volatiliraptor